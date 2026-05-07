#!/usr/bin/env python3
"""
controller.py — Orbital geometry controller

Runs a continuous loop that:
  1. Reads current satellite positions from Skyfield (real Starlink TLEs)
  2. Computes ISL and sat-to-ground one-way propagation delays
  3. Writes delays to delays.json — trgSAT.py and TN.py sleep by these
     amounts after receiving a packet, emulating orbital propagation latency
  4. Triggers Xn handover when srcSat elevation drops below threshold
     and writes the new destination to dest_override.txt

Run this BEFORE starting srcSAT.py / trgSAT.py / TN.py.

Usage:
    python3 controller.py
    python3 controller.py --list-sats        # print all satellite names in TLE file
    python3 controller.py --find-pass        # find two Starlink sats with an HO window
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

# All shared files live next to this script, regardless of where it is launched from.
_HERE = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

TLE_FILE     = str(_HERE / "starlink.tle")
SRC_SAT_NAME = "STARLINK-2657"
TGT_SAT_NAME = "STARLINK-2647"

# Ground station (USRP3 / TN node) — backhaul anchor point
GROUND_LAT_DEG =  45.5017
GROUND_LON_DEG = -73.5673
GROUND_ALT_M   =  50.0

# UE position — HO is triggered based on srcSat coverage FROM HERE, not from TN.
# Set to your actual UE location. Can be the same as ground station if needed.
UE_LAT_DEG =  45.5088     # example: slightly north of Montreal downtown
UE_LON_DEG = -73.5540
UE_ALT_M   =  10.0

# HO trigger: srcSat elevation as seen from UE drops below this angle
HO_ELEVATION_THRESHOLD_DEG = 25.0

# HO reset hysteresis: once HO fired, reset only when srcSat rises above this.
# Keeps HO from firing again until satellite is genuinely back in view.
# Set lower than HO_ELEVATION_THRESHOLD_DEG to allow repeated HOs.
HO_RESET_ELEVATION_DEG = 0.0

# How often to recompute and write delays (seconds of real time)
UPDATE_INTERVAL_S = 2.0

# Time acceleration — simulated orbital time advances this many seconds
# per real second. Use --ho-every N to set this automatically.
#   SIM_TIME_SCALE = 1    → real time (wait for actual satellite passes)
#   SIM_TIME_SCALE = 90   → HO roughly every 60 s real time
#   SIM_TIME_SCALE = 180  → HO roughly every 30 s real time
SIM_TIME_SCALE = 1.0

# Shared files read by the node scripts
DELAYS_FILE   = str(_HERE / "delays.json")
DEST_OVERRIDE = str(_HERE / "dest_override.txt")

# ══════════════════════════════════════════════════════════════════════════════


def _write_delays(isl_ms: float, gnd_ms: float):
    tmp = DELAYS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"isl_ms": round(isl_ms, 3), "gnd_ms": round(gnd_ms, 3)}, f)
    os.replace(tmp, DELAYS_FILE)


def _print_state_header(scale: float):
    tag = f"  [sim ×{scale:.0f}]" if scale != 1.0 else ""
    print(f"\n{'─'*88}")
    print(f"  {'SimTime':>8}  {'ISL km':>9}  {'ISL ms':>7}  "
          f"{'GND km':>9}  {'GND ms':>7}  {'UE→src°':>7}  {'UE→tgt°':>7}  {'HO':>4}{tag}")
    print(f"{'─'*88}")


def _find_adjacent_pairs(sats,
                         inc_tol_deg:  float = 0.5,
                         raan_tol_deg: float = 2.0,
                         ma_tol_deg:   float = 25.0) -> set[frozenset]:
    """
    Return a set of frozenset({name1, name2}) for satellite pairs that are
    orbitally adjacent: same plane (within inc/RAAN tolerance) and close in
    mean anomaly (within ma_tol_deg, accounting for wraparound).
    Adjacent pairs have ISL distances typically < 1 600 km, well below
    the sat-to-ground slant range for any non-zenith pass.
    """
    planes: dict = defaultdict(list)
    for sat in sats:
        try:
            inc  = math.degrees(sat.model.inclo)
            raan = math.degrees(sat.model.nodeo)
            ma   = math.degrees(sat.model.mo)
            # Bin by rounded inclination and RAAN to group co-planar satellites
            inc_key  = round(inc  / inc_tol_deg)
            raan_key = round(raan / raan_tol_deg)
            planes[(inc_key, raan_key)].append((ma, sat.name.strip()))
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
            dma = min(abs(ma1 - ma2), 360.0 - abs(ma1 - ma2))
            if dma <= ma_tol_deg:
                pairs.add(frozenset([n1, n2]))
        # Wraparound: last satellite neighbours first
        ma1, n1 = plane_sats[-1]
        ma2, n2 = plane_sats[0]
        dma = min(abs(ma1 - ma2), 360.0 - abs(ma1 - ma2))
        if dma <= ma_tol_deg:
            pairs.add(frozenset([n1, n2]))
    return pairs


def find_pass(tle_file: str, lat: float, lon: float, alt_m: float,
              min_el: float = 25.0, horizon_hours: int = 3,
              label: str = "ground station",
              adjacent_only: bool = False) -> None:
    """
    Scan the next `horizon_hours` hours for Starlink passes above `min_el`.
    Only shows passes that have not yet started (genuine future passes with
    full rise→set windows), sorted by rise time.
    Automatically highlights the best HO pair: two consecutive passes that
    overlap or have the shortest gap.
    """
    ts     = load.timescale()
    sats   = load.tle_file(tle_file, ts=ts)
    ground = wgs84.latlon(lat, lon, elevation_m=alt_m)

    adj_pairs: set = _find_adjacent_pairs(sats) if adjacent_only else set()

    now_unix = datetime.now(tz=timezone.utc).timestamp()
    now_tt   = ts.now()

    # Scan starts 2 minutes in the past so we catch passes currently rising,
    # but we only keep passes whose rise time is >= now (not already ending).
    start_offset_s = -120
    step_s  = 15
    n_steps = (horizon_hours * 3600 - start_offset_s) // step_s
    times   = ts.tt_jd([now_tt.tt + (start_offset_s + i * step_s) / 86400.0
                         for i in range(n_steps)])

    adj_tag = "  [adjacent-only — ISL < sat-gnd guaranteed]" if adjacent_only else ""
    print(f"\nScanning {len(sats)} satellites for passes above {min_el}° "
          f"in the next {horizon_hours}h from {label} ({lat:.3f}°N, {lon:.3f}°E)...{adj_tag}")
    print("(this may take 20–30 seconds)\n")

    passes = []   # (rise_unix, set_unix, duration_s, peak_el, sat_name)

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
                # Skip passes that have already ended or are tiny (< 60 s above threshold)
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
        r   = datetime.fromtimestamp(rise, tz=timezone.utc).strftime("%H:%M:%S")
        s   = datetime.fromtimestamp(set_,  tz=timezone.utc).strftime("%H:%M:%S")
        dur_str = f"{int(dur//60)}m{int(dur%60):02d}s"
        print(f"  {name:<22}  {r:>8}  {s:>8}  {dur_str:>8}  {peak:>7.1f}°")

    # ── Auto-suggest best HO pair ─────────────────────────────────────────────
    best_pair  = None
    best_gap_s = float("inf")

    for i in range(len(passes) - 1):
        r1, s1, _, _, n1 = passes[i]
        r2, s2, _, _, n2 = passes[i + 1]
        if adjacent_only and frozenset([n1.strip(), n2.strip()]) not in adj_pairs:
            continue
        gap = r2 - s1   # negative = overlap
        if abs(gap) < abs(best_gap_s):
            best_gap_s = gap
            best_pair  = (n1, n2, r1, s1, r2, s2, gap)

    if best_pair:
        n1, n2, r1, s1, r2, s2, gap = best_pair
        gap_str = (f"{abs(int(gap))}s overlap" if gap < 0
                   else f"{int(gap)}s gap")
        adj_note = "  [orbitally adjacent — ISL delay < sat-gnd delay]" if adjacent_only else ""
        print(f"\n  ★ Best HO pair ({gap_str}){adj_note}:")
        print(f"    SRC_SAT_NAME = \"{n1}\"")
        print(f"    TGT_SAT_NAME = \"{n2}\"")
        r1s = datetime.fromtimestamp(r1, tz=timezone.utc).strftime("%H:%M:%S")
        s1s = datetime.fromtimestamp(s1, tz=timezone.utc).strftime("%H:%M:%S")
        r2s = datetime.fromtimestamp(r2, tz=timezone.utc).strftime("%H:%M:%S")
        s2s = datetime.fromtimestamp(s2, tz=timezone.utc).strftime("%H:%M:%S")
        print(f"    srcSat visible: {r1s} → {s1s}")
        print(f"    tgtSat visible: {r2s} → {s2s}")
        print(f"\n  Copy those two names into controller.py and start before {r1s} UTC.\n")
    elif adjacent_only:
        print("\n  No adjacent-pair HO window found in this horizon. Try without --adjacent-only,")
        print("  increase --horizon-hours, or lower the min elevation threshold.\n")


def main():
    p = argparse.ArgumentParser(description="Orbital geometry controller")
    p.add_argument("--list-sats",  action="store_true",
                   help="List all satellite names in TLE file and exit")
    p.add_argument("--find-pass",      action="store_true",
                   help="Find Starlink passes over ground station and suggest HO pairs")
    p.add_argument("--adjacent-only", action="store_true",
                   help="With --find-pass: only suggest pairs in the same orbital plane "
                        "(ensures ISL delay < sat-gnd delay)")
    p.add_argument("--time-scale", type=float, default=None,
                   help="Simulated time acceleration factor (e.g. 90 → ~1 min/HO)")
    p.add_argument("--ho-every",   type=float, default=None, metavar="SECONDS",
                   help="Auto-set time scale to trigger HO approximately every N seconds")
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

    # Resolve time scale: CLI flag > --ho-every > config default
    scale = SIM_TIME_SCALE
    if args.ho_every is not None:
        scale = 5400.0 / args.ho_every   # 5400 s ≈ Starlink orbital period
        print(f"[Controller] --ho-every {args.ho_every}s → SIM_TIME_SCALE = {scale:.1f}×")
    elif args.time_scale is not None:
        scale = args.time_scale

    engine = OrbitalEngine(
        tle_path       = TLE_FILE,
        src_sat_name   = SRC_SAT_NAME,
        tgt_sat_name   = TGT_SAT_NAME,
        ground_lat_deg = GROUND_LAT_DEG,
        ground_lon_deg = GROUND_LON_DEG,
        ground_alt_m   = GROUND_ALT_M,
        ue_lat_deg     = UE_LAT_DEG,
        ue_lon_deg     = UE_LON_DEG,
        ue_alt_m       = UE_ALT_M,
    )

    running = True
    def _stop(*_):
        nonlocal running
        running = False
        print("\n[Controller] Stopping.")
    signal.signal(signal.SIGINT, _stop)

    sim_label = f"  time scale: {scale:.1f}×  (~HO every {5400/scale:.0f}s real)" if scale != 1.0 else "  time scale: real-time"
    print(f"\n[Controller] HO triggered from UE position ({UE_LAT_DEG}°N, {UE_LON_DEG}°E)")
    print(f"[Controller] HO threshold: srcSat elevation < {HO_ELEVATION_THRESHOLD_DEG}°")
    print(f"[Controller]{sim_label}")
    print(f"[Controller] Writing delays → '{DELAYS_FILE}'  Ctrl+C to stop\n")
    _print_state_header(scale)

    ho_in_progress  = False
    ho_reset_at     = None   # real monotonic time at which to force-reset (timer mode)
    # In accelerated mode, use a timer reset so HOs are periodic regardless of
    # whether srcSat happens to cross the horizon in the sim time window.
    timer_reset = (scale > 1.0)
    if timer_reset and args.ho_every:
        reset_interval_s = args.ho_every   # reset N real seconds after each HO fires
    else:
        reset_interval_s = None

    sim_time = time.time()   # simulated Unix timestamp, advances at `scale` × real speed

    while running:
        loop_start = time.monotonic()
        now_real   = time.monotonic()

        state = engine.state(unix_time=sim_time)
        _write_delays(state.isl_delay_ms, state.sat_gnd_delay_ms)

        src_el  = state.ue_src_elevation_deg
        tgt_el  = state.ue_tgt_elevation_deg
        ho_flag = ""

        # Reset: timer-based (accelerated mode) or elevation-based (real-time mode)
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
            _trigger_handover(state)

        sim_str = time.strftime("%H:%M:%S", time.gmtime(sim_time))
        print(
            f"  {sim_str}  "
            f"{state.isl_km:>9.1f}  {state.isl_delay_ms:>7.2f}  "
            f"{state.sat_gnd_km:>9.1f}  {state.sat_gnd_delay_ms:>7.2f}  "
            f"{src_el:>7.1f}  {tgt_el:>7.1f}  {ho_flag:>4}",
            flush=True,
        )

        elapsed  = time.monotonic() - loop_start
        sim_time += UPDATE_INTERVAL_S * scale
        wait      = UPDATE_INTERVAL_S - elapsed
        if wait > 0 and running:
            time.sleep(wait)


def _trigger_handover(state):
    new_dest = random.choice(["trgSAT", "TN"])   # ← replace with your algorithm

    try:
        with open(DEST_OVERRIDE, "w") as f:
            f.write(new_dest)
        dest_status = f"'{DEST_OVERRIDE}' → '{new_dest}'"
    except OSError as e:
        dest_status = f"WARNING: could not write {DEST_OVERRIDE}: {e}"

    print(f"\n{'='*80}")
    print(f"  [HO TRIGGER]  Xn Handover initiated  (sim time {time.strftime('%H:%M:%S', time.gmtime(state.timestamp))})")
    print(f"  UE→srcSat elevation = {state.ue_src_elevation_deg:.1f}°  "
          f"(below threshold {HO_ELEVATION_THRESHOLD_DEG}°)")
    print(f"  UE→tgtSat elevation = {state.ue_tgt_elevation_deg:.1f}°")
    print(f"  ISL delay           = {state.isl_delay_ms:.2f} ms")
    print(f"  Sat-gnd delay       = {state.sat_gnd_delay_ms:.2f} ms")
    print(f"  Routing override    : {dest_status}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
