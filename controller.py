#!/usr/bin/env python3
"""
controller.py — Orbital geometry controller (dynamic constellation mode)

Runs a continuous loop that:
  1. Reads current satellite positions from Skyfield (real Starlink TLEs)
  2. Computes ISL and sat-to-ground one-way propagation delays
  3. Writes delays to delays.json — trgSAT.py and TN.py sleep by these
     amounts after receiving a packet, emulating orbital propagation latency
     (or pushes them to NE-ONE via REST if --delay=NE-ONE is set)
  4. Triggers Xn handover when srcSat elevation drops below threshold;
     at that point it PROMOTES tgtSat → new srcSat, then scans the full
     constellation to pick the next best visible satellite as the new tgtSat

Dynamic mode (SRC_SAT_NAME / TGT_SAT_NAME left empty):
  The controller auto-selects the initial pair at startup (best two visible
  satellites over the UE). After every handover, the chain advances:

    STARLINK-A  →  HO fires  →  STARLINK-B serving  →  HO fires  →  STARLINK-C …

  Each new tgtSat is chosen from the full Starlink constellation: highest
  elevation above UE, guaranteed above the HO threshold, and distinct from the
  current srcSat — so it will still be visible when the next HO fires.

Fixed mode (SRC_SAT_NAME / TGT_SAT_NAME non-empty):
  Works like the original controller, but still promotes and auto-scans at each
  subsequent HO rather than cycling back to the same fixed pair.

Delay modes (--delay):
    normal  : write delays.json; trgSAT/TN apply via time.sleep()  [default]
    NE-ONE  : write delays.json AND push to NE-ONE via REST API each loop tick;
              trgSAT/TN must also be started with --delay=NE-ONE to skip sleep

Usage:
    python3 controller.py                        # dynamic auto-select, normal delays
    python3 controller.py --ho-every 30          # HO every 30 real seconds
    python3 controller.py --delay NE-ONE         # NE-ONE emulator mode
    python3 controller.py --list-sats
    python3 controller.py --find-pass
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import load, wgs84
from orbit import OrbitalEngine
from dispatcher import (
    Dispatcher, RandomDispatcher,
    AccessNode, PathScheduler,
    trgsat_bg_load, tn_bg_load,
    TASK_CYCLE,
)

_HERE = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit before running
# ══════════════════════════════════════════════════════════════════════════════

TLE_FILE = str(_HERE / "starlink.tle")

# Leave both empty to let the controller auto-select the best visible pair at
# startup and evolve the chain dynamically at each handover.
# Set both to specific names only if you want to force a particular first pair.
SRC_SAT_NAME = ""
TGT_SAT_NAME = ""

# Ground station (USRP3 / TN node) — backhaul anchor point
GROUND_LAT_DEG =  45.5017
GROUND_LON_DEG = -73.5673
GROUND_ALT_M   =  50.0

# UE position — HO is triggered based on srcSat coverage from here, not from TN.
UE_LAT_DEG =  45.5088
UE_LON_DEG = -73.5540
UE_ALT_M   =  10.0

# HO trigger: srcSat elevation as seen from UE drops below this angle
HO_ELEVATION_THRESHOLD_DEG = 25.0

# Hard cap on total HOs. Set just below the point where TLE epoch overflow
# starts producing SGP4 garbage (≈300 HOs at scale=1080× with a ~7-day-old TLE).
# Algorithm convergence saturates around HO 150–200, so 280 captures the full
# learning curve with margin and stops cleanly before bad geometry contaminates
# downstream plots / USRP latency.
HO_HARD_CAP = 280

# After HO fires, wait for srcSat to rise above this before re-arming
# (only relevant in real-time mode; accelerated mode uses a timer reset).
HO_RESET_ELEVATION_DEG = 0.0

# Real-time loop interval
UPDATE_INTERVAL_S = 2.0

# Time acceleration — simulated time advances this many seconds per real second.
# Use --ho-every N to set automatically. 1.0 = real-time.
SIM_TIME_SCALE = 1.0

DELAYS_FILE   = str(_HERE / "delays.json")
DEST_OVERRIDE = str(_HERE / "dest_override.txt")

# ── NE-ONE config — only used when running with --delay=NE-ONE ────────────────
# Set NE_ONE_HOST to the IP of your NE-ONE unit.
# NE_ONE_ISL_LINK / NE_ONE_GND_LINK are the link IDs shown in the NE-ONE GUI
# (Topology page → click a link → note the ID in the URL or properties panel).
NE_ONE_HOST     = "192.168.1.100"   # ← change to your NE-ONE IP
NE_ONE_ISL_LINK = "1"              # ← link ID for ISL  pipe in NE-ONE
NE_ONE_GND_LINK = "2"              # ← link ID for GND pipe in NE-ONE
NE_ONE_USER     = ""               # HTTP Basic auth username (leave "" if none)
NE_ONE_PASS     = ""               # HTTP Basic auth password (leave "" if none)
# ─────────────────────────────────────────────────────────────────────────────

# ── Task-type Markov switching ────────────────────────────────────────────────
# After MIN_TASK_HOS handovers, the current task type transitions according to
# this stochastic matrix.  Self-transitions keep the same type; effective mean
# session length ≈ MIN_TASK_HOS / (1 − self_prob).  Row order matches TASK_CYCLE.
#
# Design goal: near-uniform stationary distribution so all 5 task types each
# receive ~20% of HOs → every per-task PathScheduler and UPF LayerScheduler
# gets enough samples to learn.  With MIN_TASK_HOS=15 and ~120 total HOs we
# get ~8 task blocks; self-prob=0.25 gives mean block length 15/(1−0.25)=20 HOs.
#
# The slight asymmetries below reflect real traffic flow (gaming rarely jumps to
# instagram; browsing naturally transitions to youtube/mixed) while keeping the
# stationary distribution close to [0.20, 0.20, 0.20, 0.20, 0.20].
MIN_TASK_HOS = 15   # minimum HOs per task block before next transition is sampled

TASK_MARKOV: dict[str, list[float]] = {
    #              gaming  youtube  browsing  instagram  mixed
    "gaming":     [0.25,   0.20,    0.20,     0.15,     0.20],
    "youtube":    [0.15,   0.25,    0.20,     0.20,     0.20],
    "browsing":   [0.20,   0.20,    0.25,     0.15,     0.20],
    "instagram":  [0.15,   0.20,    0.20,     0.25,     0.20],
    "mixed":      [0.20,   0.20,    0.20,     0.20,     0.20],   # truly uniform
}
# ─────────────────────────────────────────────────────────────────────────────

# ── τ_max profiles (operator threshold calibration) ──────────────────────────
# 3GPP defines per-stage timeout thresholds; operators choose actual values.
# The gradient signal is  x = α(E[W] − τ_max)  per instance, so τ_max calibrates
# the algorithm's view of "fast" vs "slow":
#   x < 0  → instance favoured (grad large positive, weight grows fast)
#   x > 0  → instance penalised (grad small, weight grows slowly)
# At default values, x crosses zero around the median instance in each layer →
# clean differentiation.
#
# Three profiles for A/B testing on USRP. Run each, save dispatch_log.csv with
# a profile-tagged name, then compare convergence behaviour:
#
#   strict   τ_max ≈ ½ default  → x mostly POSITIVE → grad small
#                               → slow weight growth, weaker differentiation
#   default  current calibrated → x straddles zero  → fastest, cleanest convergence
#   relaxed  τ_max ≈ 3× default → x mostly NEGATIVE → grad SATURATES at GRAD_CAP
#                               → all instances look equally "fast" → muted gap
#
# All six per-instance τ_max values AND the path-level TAU_ACCESS switch together
# so the comparison is consistent across layers.
TAU_PROFILES: dict[str, dict[str, float]] = {
    "strict": {
        "TAU_ACCESS": 25.0,
        "ON_AMF":  0.70, "ON_SMF":  0.50, "ON_UPF":  1.00,
        "GND_AMF": 0.25, "GND_SMF": 0.15, "GND_UPF": 0.20,
    },
    "default": {
        "TAU_ACCESS": 25.0,
        "ON_AMF":  1.50, "ON_SMF":  1.00, "ON_UPF":  2.00,
        "GND_AMF": 0.50, "GND_SMF": 0.30, "GND_UPF": 0.40,
    },
    "relaxed": {
        "TAU_ACCESS": 25.0,
        "ON_AMF":  4.00, "ON_SMF":  3.00, "ON_UPF":  6.00,
        "GND_AMF": 1.50, "GND_SMF": 1.00, "GND_UPF": 1.50,
    },
}


def _apply_tau_profile(profile_name: str) -> None:
    """Patch dispatcher's per-instance τ_max + TAU_ACCESS at runtime.
    Must be called BEFORE Dispatcher() / RandomDispatcher() are constructed."""
    import dispatcher as d
    p = TAU_PROFILES[profile_name]

    d.TAU_ACCESS = p["TAU_ACCESS"]

    # τ_max is at index 7 of every spec tuple (same position in AMF/SMF/UPF specs).
    def _patch(specs: list[tuple], key: str) -> list[tuple]:
        return [s[:7] + (p[key],) + s[8:] for s in specs]

    d._ON_AMF_SPECS  = _patch(d._ON_AMF_SPECS,  "ON_AMF")
    d._ON_SMF_SPECS  = _patch(d._ON_SMF_SPECS,  "ON_SMF")
    d._ON_UPF_SPECS  = _patch(d._ON_UPF_SPECS,  "ON_UPF")
    d._GND_AMF_SPECS = _patch(d._GND_AMF_SPECS, "GND_AMF")
    d._GND_SMF_SPECS = _patch(d._GND_SMF_SPECS, "GND_SMF")
    d._GND_UPF_SPECS = _patch(d._GND_UPF_SPECS, "GND_UPF")

    print(f"\n[Controller] τ_max profile: '{profile_name}'")
    print(f"  TAU_ACCESS = {p['TAU_ACCESS']:.1f} ms")
    print(f"  ON  AMF={p['ON_AMF']:.2f}  SMF={p['ON_SMF']:.2f}  UPF={p['ON_UPF']:.2f}  (ms)")
    print(f"  GND AMF={p['GND_AMF']:.2f}  SMF={p['GND_SMF']:.2f}  UPF={p['GND_UPF']:.2f}  (ms)")

# ══════════════════════════════════════════════════════════════════════════════


def _write_delays(
    isl_ms: float, gnd_ms: float,
    src_name: str, tgt_name: str, tgt_el: float,
    task_type: str = "mixed",
):
    payload = {
        "isl_ms":     round(isl_ms, 3),
        "gnd_ms":     round(gnd_ms, 3),
        "src_sat":    src_name,
        "tgt_sat":    tgt_name,
        "tgt_el_deg": round(tgt_el, 1),
        "task_type":  task_type,
    }
    tmp = DELAYS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, DELAYS_FILE)


def _print_pair_banner(src_name: str, tgt_name: str,
                       src_el: float, tgt_el: float, isl_ms: float):
    print(f"\n[Controller] srcSat : {src_name:<22}  UE el = {src_el:>6.1f}°")
    print(f"[Controller] tgtSat : {tgt_name:<22}  UE el = {tgt_el:>6.1f}°  ISL = {isl_ms:.2f} ms")


def _print_state_header(scale: float):
    tag = f"  [sim ×{scale:.0f}]" if scale != 1.0 else ""
    print(f"\n{'─'*88}")
    print(f"  {'SimTime':>8}  {'ISL km':>9}  {'ISL ms':>7}  "
          f"{'GND km':>9}  {'GND ms':>7}  {'UE→src°':>7}  {'UE→tgt°':>7}  {'HO':>4}{tag}")
    print(f"{'─'*88}")


# ── Satellite pass scanner (for --find-pass) ──────────────────────────────────

def _find_adjacent_pairs(sats,
                         inc_tol_deg:  float = 0.5,
                         raan_tol_deg: float = 2.0,
                         ma_tol_deg:   float = 25.0) -> set[frozenset]:
    planes: dict = defaultdict(list)
    for sat in sats:
        try:
            inc  = math.degrees(sat.model.inclo)
            raan = math.degrees(sat.model.nodeo)
            ma   = math.degrees(sat.model.mo)
            planes[(round(inc / inc_tol_deg), round(raan / raan_tol_deg))].append((ma, sat.name.strip()))
        except Exception:
            continue

    pairs: set = set()
    for plane_sats in planes.values():
        if len(plane_sats) < 2:
            continue
        plane_sats.sort()
        for k in range(len(plane_sats) - 1):
            ma1, n1 = plane_sats[k]
            ma2, n2 = plane_sats[k + 1]
            if min(abs(ma1 - ma2), 360.0 - abs(ma1 - ma2)) <= ma_tol_deg:
                pairs.add(frozenset([n1, n2]))
        ma1, n1 = plane_sats[-1]
        ma2, n2 = plane_sats[0]
        if min(abs(ma1 - ma2), 360.0 - abs(ma1 - ma2)) <= ma_tol_deg:
            pairs.add(frozenset([n1, n2]))
    return pairs


def find_pass(tle_file: str, lat: float, lon: float, alt_m: float,
              min_el: float = 25.0, horizon_hours: int = 3,
              label: str = "ground station",
              adjacent_only: bool = False) -> None:
    ts     = load.timescale()
    sats   = load.tle_file(tle_file, ts=ts)
    ground = wgs84.latlon(lat, lon, elevation_m=alt_m)

    adj_pairs: set = _find_adjacent_pairs(sats) if adjacent_only else set()

    now_unix = datetime.now(tz=timezone.utc).timestamp()
    now_tt   = ts.now()

    start_offset_s = -120
    step_s  = 15
    n_steps = (horizon_hours * 3600 - start_offset_s) // step_s
    times   = ts.tt_jd([now_tt.tt + (start_offset_s + i * step_s) / 86400.0
                         for i in range(n_steps)])

    adj_tag = "  [adjacent-only — ISL < sat-gnd guaranteed]" if adjacent_only else ""
    print(f"\nScanning {len(sats)} satellites for passes above {min_el}° "
          f"in the next {horizon_hours}h from {label} ({lat:.3f}°N, {lon:.3f}°E)...{adj_tag}")
    print("(this may take 20–30 seconds)\n")

    passes = []

    for sat in sats:
        try:
            alts = (sat - ground).at(times).altaz()[0].degrees
        except Exception:
            continue

        above   = alts > min_el
        in_pass = False
        rise_i  = 0
        peak_el = 0.0

        for i, el in enumerate(alts):
            if above[i] and not in_pass:
                in_pass = True
                rise_i  = i
                peak_el = el
            elif above[i] and in_pass:
                peak_el = max(peak_el, el)
            elif not above[i] and in_pass:
                in_pass = False
                rise_unix = now_unix + start_offset_s + rise_i * step_s
                set_unix  = now_unix + start_offset_s + i      * step_s
                dur_s     = set_unix - rise_unix
                if set_unix > now_unix and dur_s >= 60:
                    passes.append((rise_unix, set_unix, dur_s, peak_el, sat.name))

    passes.sort()

    if not passes:
        print("  No passes found. Try --find-pass again in a few minutes, or lower min_el.")
        return

    print(f"  {'Satellite':<22}  {'Rise (UTC)':>8}  {'Set (UTC)':>8}  "
          f"{'Duration':>8}  {'Peak El':>8}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for rise, set_, dur, peak, name in passes[:25]:
        r       = datetime.fromtimestamp(rise, tz=timezone.utc).strftime("%H:%M:%S")
        s       = datetime.fromtimestamp(set_,  tz=timezone.utc).strftime("%H:%M:%S")
        dur_str = f"{int(dur//60)}m{int(dur%60):02d}s"
        print(f"  {name:<22}  {r:>8}  {s:>8}  {dur_str:>8}  {peak:>7.1f}°")

    best_pair  = None
    best_gap_s = float("inf")
    for i in range(len(passes) - 1):
        r1, s1, _, _, n1 = passes[i]
        r2, s2, _, _, n2 = passes[i + 1]
        if adjacent_only and frozenset([n1.strip(), n2.strip()]) not in adj_pairs:
            continue
        gap = r2 - s1
        if abs(gap) < abs(best_gap_s):
            best_gap_s = gap
            best_pair  = (n1, n2, r1, s1, r2, s2, gap)

    if best_pair:
        n1, n2, r1, s1, r2, s2, gap = best_pair
        gap_str  = f"{abs(int(gap))}s overlap" if gap < 0 else f"{int(gap)}s gap"
        adj_note = "  [orbitally adjacent]" if adjacent_only else ""
        print(f"\n  ★ Best HO pair ({gap_str}){adj_note}:")
        print(f"    SRC_SAT_NAME = \"{n1}\"")
        print(f"    TGT_SAT_NAME = \"{n2}\"")
        r1s = datetime.fromtimestamp(r1, tz=timezone.utc).strftime("%H:%M:%S")
        s1s = datetime.fromtimestamp(s1, tz=timezone.utc).strftime("%H:%M:%S")
        r2s = datetime.fromtimestamp(r2, tz=timezone.utc).strftime("%H:%M:%S")
        s2s = datetime.fromtimestamp(s2, tz=timezone.utc).strftime("%H:%M:%S")
        print(f"    srcSat visible: {r1s} → {s1s}")
        print(f"    tgtSat visible: {r2s} → {s2s}")
        print(f"\n  Copy those two names into SRC_SAT_NAME / TGT_SAT_NAME if you want a fixed "
              f"first pair, or leave them empty for full auto-selection.\n")
    elif adjacent_only:
        print("\n  No adjacent-pair HO window found. Try without --adjacent-only or increase "
              "--horizon-hours.\n")


# ── Handover trigger ──────────────────────────────────────────────────────────

def _trigger_handover(
    state,
    engine:           OrbitalEngine,
    sim_time:         float,
    min_el:           float,
    dispatcher:       Dispatcher,
    trgsat_node:      AccessNode,
    tn_node:          AccessNode,
    task_type:        str,
    path_schedulers:  "dict[str, PathScheduler]",
    rand_dispatcher:  "RandomDispatcher | None" = None,
) -> bool:
    """
    Execute an Xn handover via two-level Bregman learning:
      1. Compute access costs for both paths
      2. Level-1: per-task-type PathScheduler samples path (Bregman mirror descent)
      3. Level-2: Dispatcher selects AMF/SMF/UPF instances (Bregman)
      4. PathScheduler updated with BOTH access costs (full-information Bregman)
      5. RandomDispatcher: 50/50 random path + uniform NF (baseline)
      6. Write dest_override.txt, promote tgtSat, scan for next tgtSat

    Returns True if next tgt satellite was found, False otherwise.
    """
    isl_ms = state.isl_delay_ms
    gnd_ms = state.sat_gnd_delay_ms

    # ── Physical bounds guard — reject SGP4 garbage from TLE epoch overflow ──
    # LEO ISL physical max ≈ 8.3 ms (2500 km); LEO-GND physical max ≈ 3.7 ms (1123 km).
    # Anything above these generous thresholds means TLE validity window exceeded.
    _MAX_ISL_MS = 100.0
    _MAX_GND_MS = 50.0
    if isl_ms > _MAX_ISL_MS or gnd_ms > _MAX_GND_MS:
        print(
            f"  [HO SKIPPED] Propagation delay out of physical bounds: "
            f"ISL={isl_ms:.1f} ms  GND={gnd_ms:.1f} ms  "
            f"(TLE epoch likely exceeded — recommend restarting simulation or "
            f"capping at ≤150 HOs with the current starlink.tle)"
        )
        return False
    # ─────────────────────────────────────────────────────────────────────────

    # ── Level-1: access costs + per-task Bregman PathScheduler ──────────────
    cost_isl = trgsat_node.total_access_cost_ms(isl_ms)
    cost_gnd = tn_node.total_access_cost_ms(gnd_ms)

    sched = path_schedulers[task_type]
    path  = sched.sample()

    # ── Level-2: Bregman NF instance selection ────────────────────────────────
    _, info = dispatcher.dispatch(
        path            = path,
        isl_ms          = isl_ms,
        gnd_ms          = gnd_ms,
        task_type       = task_type,
        path_p_isl      = sched.p_isl,
        path_p_gnd      = sched.p_gnd,
        access_cost_isl = cost_isl,
        access_cost_gnd = cost_gnd,
        trgsat_bg       = trgsat_node.bg_load,
        tn_bg           = tn_node.bg_load,
    )

    # Full-information Bregman update: total cost (access + best reachable compute)
    # PathScheduler must see compute cost so it learns task-specific path preferences.
    # e.g. instagram: all ON-UPF unstable → ISL total >> GND total → correct GND bias.
    exp_isl = dispatcher.expected_compute_ms("ISL", task_type)
    exp_gnd = dispatcher.expected_compute_ms("GND", task_type)
    sched.update(cost_isl + exp_isl, cost_gnd + exp_gnd)

    # ── Random baseline: 50/50 path + uniform NF (independent of Bregman) ────
    if rand_dispatcher is not None:
        rand_dispatcher.dispatch(
            isl_ms          = isl_ms,
            gnd_ms          = gnd_ms,
            task_type       = task_type,
            access_cost_isl = cost_isl,
            access_cost_gnd = cost_gnd,
            trgsat_bg       = trgsat_node.bg_load,
            tn_bg           = tn_node.bg_load,
        )

    new_dest = "trgSAT" if path == "ISL" else "TN"

    try:
        with open(DEST_OVERRIDE, "w") as f:
            f.write(new_dest)
        dest_status = f"'{DEST_OVERRIDE}' → '{new_dest}'  [{path} path]"
    except OSError as e:
        dest_status = f"WARNING: could not write {DEST_OVERRIDE}: {e}"

    old_src = engine.src_sat
    new_src = engine.tgt_sat   # tgt becomes the new serving satellite

    print(f"\n{'='*80}")
    print(f"  [HO TRIGGER]  Xn Handover initiated  "
          f"(sim time {time.strftime('%H:%M:%S', time.gmtime(state.timestamp))})")
    print(f"  Task type  : {task_type}")
    print(f"  UE→srcSat ({state.src_name}) elevation = {state.ue_src_elevation_deg:.1f}°  "
          f"(below threshold {min_el}°)")
    print(f"  UE→tgtSat ({state.tgt_name}) elevation = {state.ue_tgt_elevation_deg:.1f}°")
    print(f"  Prop delay : ISL={isl_ms:.2f} ms  |  GND={gnd_ms:.2f} ms")
    print(f"  TrgSAT bg={trgsat_node.bg_load:.2f}  "
          f"total_ISL={cost_isl + exp_isl:.2f} ms (acc={cost_isl:.2f}+cmp={exp_isl:.2f})"
          f"  →  π_ISL[{task_type}]={sched.p_isl:.3f}")
    print(f"  TN     bg={tn_node.bg_load:.2f}  "
          f"total_GND={cost_gnd + exp_gnd:.2f} ms (acc={cost_gnd:.2f}+cmp={exp_gnd:.2f})"
          f"  →  π_GND[{task_type}]={sched.p_gnd:.3f}")
    print(f"  Path selected  : {path}")
    print(f"  Core delay : AMF={info['amf_ms']:.2f} ms  SMF={info['smf_ms']:.2f} ms  "
          f"UPF={info['upf_ms']:.2f} ms  →  total={info['total_ms']:.2f} ms")
    print(f"  Regret     : access={info['access_regret_ms']:.2f} ms  "
          f"inst={info['inst_regret_ms']:.2f} ms  "
          f"global={info['global_regret_ms']:.2f} ms  "
          f"(cum_global={info['cum_global_regret_ms']:.0f} ms)")
    print(f"  Routing    : {dest_status}")
    print(f"  Promoting  : {state.tgt_name} → new srcSat  |  "
          f"{state.src_name} → retired")
    print(f"  Scanning full constellation for next tgtSat "
          f"(el > {min_el}°, excluding current pair)…", flush=True)

    t_scan_start = time.monotonic()
    new_tgt = engine.best_visible_sat(
        unix_time     = sim_time,
        exclude_names = {state.src_name, state.tgt_name},
        min_el        = min_el,
    )
    scan_s = time.monotonic() - t_scan_start

    if new_tgt is None:
        print(f"  WARNING: No visible satellite found above {min_el}° "
              f"(scan took {scan_s:.1f}s). Keeping promoted tgt as src, "
              f"tgt temporarily set to old src so ISL remains defined.")
        engine.update_pair(new_src, old_src)
        print(f"{'='*80}\n")
        return False

    new_tgt_sat, new_tgt_el = new_tgt
    engine.update_pair(new_src, new_tgt_sat)

    print(f"  ✓ Next tgtSat : {new_tgt_sat.name.strip():<22}  "
          f"UE el = {new_tgt_el:.1f}°  (scan took {scan_s:.1f}s)")
    print(f"{'='*80}\n")
    return True


# ── Main control loop ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Orbital geometry controller (dynamic mode)")
    p.add_argument("--list-sats",    action="store_true",
                   help="List all satellite names in TLE file and exit")
    p.add_argument("--find-pass",    action="store_true",
                   help="Find Starlink passes over UE and suggest HO pairs")
    p.add_argument("--adjacent-only", action="store_true",
                   help="With --find-pass: only suggest same-plane adjacent pairs")
    p.add_argument("--time-scale",   type=float, default=None,
                   help="Simulated time acceleration factor")
    p.add_argument("--ho-every",     type=float, default=None, metavar="SECONDS",
                   help="Auto-set time scale to trigger HO approximately every N real seconds")
    p.add_argument("--delay",        choices=["normal", "NE-ONE"], default="normal",
                   help="Delay mode: 'normal'=delays.json+receiver sleep (default), "
                        "'NE-ONE'=also push delays to NE-ONE emulator via REST API")
    p.add_argument("--tau-profile",  choices=list(TAU_PROFILES.keys()), default="default",
                   help="τ_max profile: strict | default | relaxed.  Controls how "
                        "tightly the algorithm's gradient signal is calibrated against "
                        "each NF instance's expected delay.  Useful for USRP A/B testing.")
    args = p.parse_args()

    if args.list_sats:
        ts   = load.timescale()
        sats = load.tle_file(TLE_FILE, ts=ts)
        print(f"\nSatellites in {TLE_FILE}:\n")
        for s in sorted(sats, key=lambda x: x.name):
            print(f"  {s.name}")
        print()
        return

    if args.find_pass:
        find_pass(TLE_FILE, UE_LAT_DEG, UE_LON_DEG, UE_ALT_M,
                  min_el=HO_ELEVATION_THRESHOLD_DEG, label="UE",
                  adjacent_only=args.adjacent_only)
        return

    scale = SIM_TIME_SCALE
    if args.ho_every is not None:
        scale = 5400.0 / args.ho_every
        print(f"[Controller] --ho-every {args.ho_every}s → SIM_TIME_SCALE = {scale:.1f}×")
    elif args.time_scale is not None:
        scale = args.time_scale

    engine = OrbitalEngine(
        tle_path       = TLE_FILE,
        ground_lat_deg = GROUND_LAT_DEG,
        ground_lon_deg = GROUND_LON_DEG,
        ground_alt_m   = GROUND_ALT_M,
        ue_lat_deg     = UE_LAT_DEG,
        ue_lon_deg     = UE_LON_DEG,
        ue_alt_m       = UE_ALT_M,
        src_sat_name   = SRC_SAT_NAME,
        tgt_sat_name   = TGT_SAT_NAME,
    )

    sim_time = time.time()   # simulated Unix timestamp

    # ── Auto-select initial pair if names not configured ──────────────────────
    if engine.src_sat is None or engine.tgt_sat is None:
        print(f"\n[Controller] Auto-selecting initial src/tgt pair "
              f"(scanning {len(engine._all_sats)} satellites, please wait…)", flush=True)
        t0 = time.monotonic()
        src, tgt = engine.two_best_visible(sim_time, min_el=HO_ELEVATION_THRESHOLD_DEG)
        elapsed = time.monotonic() - t0

        if src is None:
            print(f"[Controller] ERROR: No satellites visible above "
                  f"{HO_ELEVATION_THRESHOLD_DEG}° from UE. "
                  f"Check UE_LAT_DEG / UE_LON_DEG in config.")
            return
        if tgt is None:
            print(f"[Controller] WARNING: Only one satellite visible above threshold.")
            tgt = src  # degenerate case; ISL will be 0

        engine.update_pair(src, tgt)
        print(f"[Controller] Scan complete in {elapsed:.1f}s")

    # Apply τ_max profile BEFORE Dispatcher init (it reads dispatcher specs once).
    _apply_tau_profile(args.tau_profile)

    dispatcher      = Dispatcher(log_dir=_HERE, tag=args.tau_profile)
    rand_dispatcher = RandomDispatcher(log_dir=_HERE, tag=args.tau_profile)
    # One PathScheduler per task type — each learns its own π_ISL / π_GND
    path_schedulers: dict[str, PathScheduler] = {tt: PathScheduler() for tt in TASK_CYCLE}

    # ── Access nodes (srcSAT decides path based on Xn setup cost) ────────────
    # Physics-correct ordering: ISL single-hop is FASTER baseline than GND multi-hop.
    # TrgSAT (ISL): xn_base=3.0 ms — single satellite-to-satellite hop, lightweight
    #               Xn protocol.  bg_load ∈ [0.20, 0.85]: limited onboard capacity,
    #               so M/M/1 sojourn 1/(1−ρ) inflates sharply near peak load.
    # TN     (GND): xn_base=5.0 ms — multi-hop ground core signaling (gNB → AMF →
    #               anchor UPF).  bg_load ∈ [0.10, 0.60]: elastic capacity, mild ρ.
    #
    # Crossover (access only): bg_isl ≈ 0.82 with mid bg_tn = 0.35 (top of the
    # ISL bg cycle).  Compute pulls instagram's transition lower (≈0.70) because
    # ISL compute is ~4 ms (UPF-ON-2 only) vs ~0.7 ms on GND.
    trgsat_node = AccessNode(name="TrgSAT", ngap_ms=1.0, xn_base_ms=4.0)
    tn_node     = AccessNode(name="TN",     ngap_ms=0.5, xn_base_ms=5.0)

    ho_count      = 0            # total HOs fired
    task_type     = random.choice(TASK_CYCLE)   # random start — no single task gets a head start
    task_ho_count = 0            # HOs elapsed since last task transition

    # ── NE-ONE setup ──────────────────────────────────────────────────────────
    neone = None
    if args.delay == "NE-ONE":
        from neone_ctrl import NeoNEController
        neone = NeoNEController(
            host        = NE_ONE_HOST,
            isl_link_id = NE_ONE_ISL_LINK,
            gnd_link_id = NE_ONE_GND_LINK,
            username    = NE_ONE_USER,
            password    = NE_ONE_PASS,
        )
        print(f"[Controller] Delay mode : NE-ONE  "
              f"(delays pushed to {NE_ONE_HOST} each loop tick)")
        print(f"[Controller] trgSAT.py and TN.py must also be started with --delay=NE-ONE")
    else:
        print(f"[Controller] Delay mode : normal  "
              f"(delays.json written; receivers apply via sleep)")

    running = True
    def _stop(*_):
        nonlocal running
        running = False
        print("\n[Controller] Stopping.")
        dispatcher.close()
        rand_dispatcher.close()
    signal.signal(signal.SIGINT, _stop)

    sim_label = (f"  time scale: {scale:.1f}×  (~HO every {5400/scale:.0f}s real)"
                 if scale != 1.0 else "  time scale: real-time")
    print(f"\n[Controller] HO threshold: srcSat UE elevation < {HO_ELEVATION_THRESHOLD_DEG}°")
    print(f"[Controller]{sim_label}")
    print(f"[Controller] Writing delays → '{DELAYS_FILE}'  Ctrl+C to stop")

    # Print the initial pair banner before the table header
    init_state = engine.state(unix_time=sim_time)
    _print_pair_banner(
        init_state.src_name, init_state.tgt_name,
        init_state.ue_src_elevation_deg, init_state.ue_tgt_elevation_deg,
        init_state.isl_delay_ms,
    )
    _print_state_header(scale)

    ho_in_progress = False
    ho_reset_at    = None
    timer_reset    = (scale > 1.0)
    reset_interval_s = args.ho_every if (timer_reset and args.ho_every) else None

    # ── delays.json sanity guard ──────────────────────────────────────────────
    # LEO geometry caps the physically possible values; anything above means
    # SGP4 has returned garbage (stale TLE / epoch overflow). When a tick is
    # out of bounds we re-use the last valid pair so delays.json never
    # contains nonsense values that would cause trgSAT/TN to sleep for seconds.
    _DELAY_MAX_ISL_MS = 100.0   # LEO sat-to-sat ≤ ~57 ms (17 000 km)
    _DELAY_MAX_GND_MS =  50.0   # LEO sat-to-ground ≤ ~3.7 ms at zenith, ≤ ~20 ms at horizon
    last_valid_isl_ms = None
    last_valid_gnd_ms = None
    bad_tick_warned   = False

    while running:
        loop_start = time.monotonic()
        now_real   = time.monotonic()

        state = engine.state(unix_time=sim_time)

        isl_for_write = state.isl_delay_ms
        gnd_for_write = state.sat_gnd_delay_ms
        if isl_for_write > _DELAY_MAX_ISL_MS or gnd_for_write > _DELAY_MAX_GND_MS:
            # SGP4 garbage — fall back to last valid pair (if any).
            if last_valid_isl_ms is not None:
                isl_for_write = last_valid_isl_ms
                gnd_for_write = last_valid_gnd_ms
            if not bad_tick_warned:
                print(f"  [WARN] SGP4 out of bounds  ISL={state.isl_delay_ms:.0f}ms  "
                      f"GND={state.sat_gnd_delay_ms:.0f}ms  → using last valid values. "
                      f"Refresh starlink.tle or restart simulation.", flush=True)
                bad_tick_warned = True
        else:
            last_valid_isl_ms = isl_for_write
            last_valid_gnd_ms = gnd_for_write
            bad_tick_warned   = False

        _write_delays(
            isl_for_write, gnd_for_write,
            state.src_name, state.tgt_name, state.ue_tgt_elevation_deg,
            task_type=task_type,
        )
        if neone is not None:
            neone.set_both(isl_for_write, gnd_for_write)

        src_el  = state.ue_src_elevation_deg
        ho_flag = ""

        if ho_in_progress:
            if timer_reset and ho_reset_at is not None and now_real >= ho_reset_at:
                ho_in_progress = False
                ho_reset_at    = None
            elif not timer_reset and src_el >= HO_RESET_ELEVATION_DEG:
                ho_in_progress = False

        if src_el < HO_ELEVATION_THRESHOLD_DEG and not ho_in_progress:
            ho_in_progress = True
            ho_flag = "HO!"
            if reset_interval_s:
                ho_reset_at = now_real + reset_interval_s

            # Update background loads for this HO.
            trgsat_node.bg_load = trgsat_bg_load(ho_count)
            tn_node.bg_load     = tn_bg_load(ho_count)
            ho_count       += 1
            task_ho_count  += 1

            # ── Hard cap: stop cleanly before TLE epoch overflow corrupts data ──
            if ho_count >= HO_HARD_CAP:
                print(f"\n{'='*80}")
                print(f"  [HO HARD CAP REACHED]  {ho_count} handovers completed.")
                print(f"  Algorithm has fully converged (saturates around HO ~150-200).")
                print(f"  Stopping cleanly before SGP4 garbage from stale TLE epoch.")
                print(f"  Refresh starlink.tle to extend the simulation window.")
                print(f"{'='*80}\n", flush=True)
                running = False
                dispatcher.close()
                rand_dispatcher.close()
                break

            # Markov task switching: after MIN_TASK_HOS HOs sample the next task.
            # Self-transitions keep the current type; the minimum block of
            # MIN_TASK_HOS HOs ensures each per-task scheduler sees enough data
            # before the environment can shift.
            if task_ho_count >= MIN_TASK_HOS:
                prev_task     = task_type
                task_type     = random.choices(TASK_CYCLE,
                                               weights=TASK_MARKOV[task_type])[0]
                task_ho_count = 0
                if task_type != prev_task:
                    print(f"  [Task switch]  {prev_task} → {task_type}  "
                          f"(after {MIN_TASK_HOS} HOs)", flush=True)

            _trigger_handover(state, engine, sim_time, HO_ELEVATION_THRESHOLD_DEG,
                              dispatcher, trgsat_node, tn_node, task_type,
                              path_schedulers, rand_dispatcher)

            # Recompute state with the new pair for accurate display
            state = engine.state(unix_time=sim_time)
            _print_pair_banner(
                state.src_name, state.tgt_name,
                state.ue_src_elevation_deg, state.ue_tgt_elevation_deg,
                state.isl_delay_ms,
            )
            _print_state_header(scale)

        sim_str = time.strftime("%H:%M:%S", time.gmtime(sim_time))
        print(
            f"  {sim_str}  "
            f"{state.isl_km:>9.1f}  {state.isl_delay_ms:>7.2f}  "
            f"{state.sat_gnd_km:>9.1f}  {state.sat_gnd_delay_ms:>7.2f}  "
            f"{state.ue_src_elevation_deg:>7.1f}  {state.ue_tgt_elevation_deg:>7.1f}  "
            f"{ho_flag:>4}",
            flush=True,
        )

        elapsed  = time.monotonic() - loop_start
        sim_time += UPDATE_INTERVAL_S * scale
        wait      = UPDATE_INTERVAL_S - elapsed
        if wait > 0 and running:
            time.sleep(wait)


if __name__ == "__main__":
    main()
