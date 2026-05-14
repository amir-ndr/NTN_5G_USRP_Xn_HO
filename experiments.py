#!/usr/bin/env python3
"""
experiments.py — Paper experimental section: two figures from real Starlink TLE.

  Figure 1 (experiments_summary.png) — two summary panels:
    (A) Per-task path differentiation under uniform task-load multipliers
        m ∈ {0.5, 1.0, 1.5, 2.0, 2.5}.  Tasks cross to GND in CPR order:
          instagram (m≈1.43) → mixed (1.78) → browsing (1.90) → youtube (2.44).
        Gaming stays on ISL because its CPR is the lowest.
    (B) Path-layer response to satellite signaling congestion
        TRGSAT_BG_PEAK ∈ {0.50, 0.70, 0.85, 0.95}.

  Figure 2 (experiments_probs.png) — selection probability vs HO index for
    AMF / SMF / UPF on BOTH paths (ISL and GND), at THREE task-load levels:
      light (m=0.5), default (m=1.0), heavy (m=2.0).
    Demonstrates that the algorithm's per-layer selection behaviour changes
    correctly as the load changes — at heavy load, unstable ON-UPF instances
    are correctly avoided and weight concentrates on UPF-ON-2 (or shifts to
    GND if all ON-UPF infeasible).

Prop delays come from real Starlink TLE (Montreal UE/TN, same as controller.py).
Precomputed once, cached, reused across every sweep point.

USRP not used here (passive consumer; would multiply runtime ~100×).
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dispatcher as d
from orbit import OrbitalEngine

_HERE = Path(__file__).resolve().parent
OUT_SUMMARY = _HERE / "experiments_summary.png"
OUT_PROBS   = _HERE / "experiments_probs.png"
DELAYS_CACHE = _HERE / "experiments_delays.npy"
TLE_FILE     = _HERE / "starlink.tle"

UE_LAT, UE_LON, UE_ALT = 45.5088, -73.5540, 10.0
GS_LAT, GS_LON, GS_ALT = 45.5017, -73.5673, 50.0

# ── Sweep configuration ──────────────────────────────────────────────────────
N_HO = 400
SEED = 42

LOAD_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5]            # Figure 1 panel A
BG_PEAK_VALUES   = [0.50, 0.70, 0.85, 0.95]             # Figure 1 panel B
PROBS_LOADS      = [0.5, 1.0, 2.0]                      # Figure 2 (3 rows)
FOCUS_TASK       = "instagram"                          # Figure 2 focus


# ═══════════════════════════════════════════════════════════════════════════
#  Realistic prop-delay precomputation
# ═══════════════════════════════════════════════════════════════════════════

def precompute_realistic_delays(n_ho: int, sim_step_s: float = 30.0) -> np.ndarray:
    """Real Starlink TLE → (isl_ms, gnd_ms) trajectory. Refreshes pair on the
    same trigger conditions as the live controller (src elev < 25° or tgt
    sets below horizon), so the trajectory mirrors realistic Xn handover
    geometry."""
    print(f"  Precomputing {n_ho} prop-delay samples from real Starlink TLE...",
          flush=True)
    eng = OrbitalEngine(
        tle_path       = str(TLE_FILE),
        ground_lat_deg = GS_LAT, ground_lon_deg = GS_LON, ground_alt_m = GS_ALT,
        ue_lat_deg     = UE_LAT, ue_lon_deg     = UE_LON, ue_alt_m     = UE_ALT,
    )
    sim_time = time.time()
    src, tgt = eng.two_best_visible(sim_time, min_el=25.0)
    if src is None or tgt is None:
        raise RuntimeError("No visible satellites at current time. Refresh starlink.tle.")
    eng.update_pair(src, tgt)

    # Refresh policy: only when geometry truly breaks (TLE garbage, tgt sets,
    # or ISL exceeds physical LEO max). This lets the pair drift across the sky
    # so ISL takes natural values from a few hundred km up to several thousand,
    # producing realistic ISL ranges (5–30 ms) without over-refreshing.
    delays = np.zeros((n_ho, 2))
    refreshes = 0
    last_valid = (18.0, 29.0)
    t0 = time.monotonic()

    for i in range(n_ho):
        state = eng.state(unix_time=sim_time)
        isl, gnd = state.isl_delay_ms, state.sat_gnd_delay_ms

        need_refresh = (
            not (0.5 < isl < 50.0 and 1.0 < gnd < 30.0)   # TLE/garbage guard
            or state.ue_tgt_elevation_deg < -5.0          # tgt clearly set
        )
        if need_refresh:
            new_pair = eng.two_best_visible(sim_time, min_el=25.0)
            if new_pair[0] is not None and new_pair[1] is not None:
                eng.update_pair(*new_pair)
                state = eng.state(unix_time=sim_time)
                isl, gnd = state.isl_delay_ms, state.sat_gnd_delay_ms
                refreshes += 1

        if 0.5 < isl < 50.0 and 1.0 < gnd < 30.0:
            last_valid = (isl, gnd)
        delays[i] = last_valid
        sim_time += sim_step_s

    elapsed = time.monotonic() - t0
    print(f"  done in {elapsed:.1f}s  (covers {n_ho*sim_step_s/3600:.1f} h sim time, "
          f"{refreshes} pair refreshes)")
    print(f"  ISL range = [{delays[:,0].min():.1f}, {delays[:,0].max():.1f}] ms, "
          f"GND range = [{delays[:,1].min():.1f}, {delays[:,1].max():.1f}] ms")
    return delays


def get_delays(n_ho: int, force_rebuild: bool = False) -> np.ndarray:
    if not force_rebuild and DELAYS_CACHE.exists():
        cached = np.load(DELAYS_CACHE)
        if len(cached) >= n_ho:
            print(f"  Using cached delays (n={n_ho}, "
                  f"ISL_avg={cached[:n_ho,0].mean():.2f} ms, "
                  f"GND_avg={cached[:n_ho,1].mean():.2f} ms)")
            return cached[:n_ho]
    arr = precompute_realistic_delays(n_ho)
    np.save(DELAYS_CACHE, arr)
    return arr


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation core — returns either summary stats or full prob traces
# ═══════════════════════════════════════════════════════════════════════════

def _run_one(real_delays: np.ndarray, *, record_traces: bool = False,
             focus_task: str = FOCUS_TASK, seed: int = SEED) -> dict:
    """Run one full simulation. If record_traces=True, also records per-HO
    selection probabilities for AMF/SMF/UPF on both paths (focus_task only
    for UPF since UPF is per-task)."""
    n_ho = len(real_delays)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    block = n_ho // len(d.TASK_CYCLE)
    task_seq: list[str] = []
    for tt in d.TASK_CYCLE:
        task_seq.extend([tt] * block)
    while len(task_seq) < n_ho:
        task_seq.append(d.TASK_CYCLE[-1])

    traces = None
    if record_traces:
        traces = {
            "amf_on":   [[] for _ in range(3)],
            "smf_on":   [[] for _ in range(3)],
            "upf_on":   [[] for _ in range(3)],
            "amf_gnd":  [[] for _ in range(3)],
            "smf_gnd":  [[] for _ in range(3)],
            "upf_gnd":  [[] for _ in range(3)],
        }

    with tempfile.TemporaryDirectory() as tmp:
        disp        = d.Dispatcher(tmp)
        path_scheds = {tt: d.PathScheduler() for tt in d.TASK_CYCLE}
        trgsat      = d.AccessNode("TrgSAT", ngap_ms=1.0, xn_base_ms=3.0)
        tn          = d.AccessNode("TN",     ngap_ms=0.5, xn_base_ms=5.0)

        path_count = {tt: {"ISL": 0, "GND": 0} for tt in d.TASK_CYCLE}

        for ho_id in range(n_ho):
            task = task_seq[ho_id]
            trgsat.bg_load = d.trgsat_bg_load(ho_id)
            tn.bg_load     = d.tn_bg_load(ho_id)

            isl_ms = float(real_delays[ho_id, 0])
            gnd_ms = float(real_delays[ho_id, 1])
            cost_isl = trgsat.total_access_cost_ms(isl_ms)
            cost_gnd = tn.total_access_cost_ms(gnd_ms)

            sched = path_scheds[task]
            path  = sched.sample()
            disp.dispatch(
                path=path, isl_ms=isl_ms, gnd_ms=gnd_ms, task_type=task,
                path_p_isl=sched.p_isl, path_p_gnd=sched.p_gnd,
                access_cost_isl=cost_isl, access_cost_gnd=cost_gnd,
                trgsat_bg=trgsat.bg_load, tn_bg=tn.bg_load,
            )
            exp_isl = disp.expected_compute_ms("ISL", task)
            exp_gnd = disp.expected_compute_ms("GND", task)
            sched.update(cost_isl + exp_isl, cost_gnd + exp_gnd)
            path_count[task][path] += 1

            if record_traces:
                amf_on  = disp.on_amf.probabilities()
                smf_on  = disp.on_smf.probabilities()
                upf_on  = disp.on_upf[focus_task].probabilities()
                amf_gnd = disp.gnd_amf.probabilities()
                smf_gnd = disp.gnd_smf.probabilities()
                upf_gnd = disp.gnd_upf[focus_task].probabilities()
                for i in range(3):
                    traces["amf_on"][i].append(amf_on[i])
                    traces["smf_on"][i].append(smf_on[i])
                    traces["upf_on"][i].append(upf_on[i])
                    traces["amf_gnd"][i].append(amf_gnd[i])
                    traces["smf_gnd"][i].append(smf_gnd[i])
                    traces["upf_gnd"][i].append(upf_gnd[i])

        per_task_gnd_frac = {
            tt: path_count[tt]["GND"] / max(sum(path_count[tt].values()), 1)
            for tt in d.TASK_CYCLE
        }
        overall_gnd_frac = sum(v["GND"] for v in path_count.values()) / max(
            sum(sum(v.values()) for v in path_count.values()), 1
        )
        disp.close()

    return {
        "per_task_gnd_frac": per_task_gnd_frac,
        "overall_gnd_frac":  overall_gnd_frac,
        "traces":            traces,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Sweep functions
# ═══════════════════════════════════════════════════════════════════════════

def sweep_task_load(delays: np.ndarray) -> list[dict]:
    saved = dict(d.TASK_N_TASKS)
    out = []
    print("\n[Sweep A] Uniform task-load multiplier")
    for m in LOAD_MULTIPLIERS:
        for tt, base in saved.items():
            d.TASK_N_TASKS[tt] = int(round(base * m))
        r = _run_one(delays)
        r["multiplier"] = m
        out.append(r)
        s = " ".join(f"{tt[:3]}={r['per_task_gnd_frac'][tt]*100:4.0f}%"
                      for tt in d.TASK_CYCLE)
        print(f"  m={m:.1f}: {s}")
    d.TASK_N_TASKS.clear()
    d.TASK_N_TASKS.update(saved)
    return out


def sweep_bg_peak(delays: np.ndarray) -> list[dict]:
    saved = d.TRGSAT_BG_PEAK
    out = []
    print("\n[Sweep C] TRGSAT_BG_PEAK")
    for peak in BG_PEAK_VALUES:
        d.TRGSAT_BG_PEAK = peak
        r = _run_one(delays)
        r["bg_peak"] = peak
        out.append(r)
        print(f"  BG_PEAK={peak:.2f}: overall %GND={r['overall_gnd_frac']*100:5.1f}%")
    d.TRGSAT_BG_PEAK = saved
    return out


def collect_traces_at_loads(delays: np.ndarray) -> dict:
    """For each load multiplier in PROBS_LOADS, run with record_traces=True
    and return {m: traces_dict}."""
    saved = dict(d.TASK_N_TASKS)
    all_traces = {}
    print("\n[Probability traces] Recording per-HO selection probabilities "
          f"for task='{FOCUS_TASK}' at 3 load levels")
    for m in PROBS_LOADS:
        for tt, base in saved.items():
            d.TASK_N_TASKS[tt] = int(round(base * m))
        r = _run_one(delays, record_traces=True, focus_task=FOCUS_TASK)
        all_traces[m] = r["traces"]
        # Compute final probability concentrations for quick reporting
        final_amf_on = [r["traces"]["amf_on"][i][-1] for i in range(3)]
        final_upf_on = [r["traces"]["upf_on"][i][-1] for i in range(3)]
        print(f"  m={m:.1f}: final AMF-ON probs={[f'{p:.2f}' for p in final_amf_on]}  "
              f"UPF-ON probs={[f'{p:.2f}' for p in final_upf_on]}")
    d.TASK_N_TASKS.clear()
    d.TASK_N_TASKS.update(saved)
    return all_traces


# ═══════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════

TASK_COLORS = {
    "gaming":    "#2ca02c", "youtube":   "#d62728", "browsing":  "#9467bd",
    "instagram": "#ff7f0e", "mixed":     "#1f77b4",
}
LOAD_CRIT = {"instagram": 1.43, "mixed": 1.78, "browsing": 1.90,
             "youtube": 2.44, "gaming": 8.55}

# Colors for instance index 0/1/2 (consistent across panels)
INST_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
INST_STYLES = ["-", "--", "-."]


def make_summary_figure(sa: list[dict], sc: list[dict],
                        delays_summary: str) -> None:
    fig, (axA, axC) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Algorithm Behaviour — Trend Summary  "
                 f"(real Starlink TLE: {delays_summary}, N={N_HO} HOs per point)",
                 fontsize=12, fontweight="bold")

    # Panel A — task differentiation under uniform load
    ms = [r["multiplier"] for r in sa]
    for tt in d.TASK_CYCLE:
        y = [r["per_task_gnd_frac"][tt] * 100 for r in sa]
        axA.plot(ms, y, "o-", lw=2, ms=8, color=TASK_COLORS[tt],
                 label=f"{tt}  (crit. m={LOAD_CRIT[tt]:.2f})")
    axA.set_xlabel("Uniform task-load multiplier  m", fontsize=11)
    axA.set_ylabel("% HOs routed to GND", fontsize=11)
    axA.set_title("(A) Global load ↑ → tasks cross to GND in CPR order")
    axA.set_xticks(ms)
    axA.set_ylim(-3, 105)
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="upper left",
                title="task  (analytical crit. multiplier)")

    # Panel C — satellite congestion
    peaks = [r["bg_peak"] for r in sc]
    gnd   = [r["overall_gnd_frac"] * 100 for r in sc]
    axC.plot(peaks, gnd, "o-", lw=2.5, ms=10, color="#1f77b4")
    axC.fill_between(peaks, 0, gnd, alpha=0.15, color="#1f77b4")
    for x, y in zip(peaks, gnd):
        axC.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    axC.set_xlabel("TRGSAT_BG_PEAK  (sat Xn peak utilization)", fontsize=11)
    axC.set_ylabel("Overall % HOs routed to GND", fontsize=11)
    axC.set_title("(B) Satellite congestion ↑ → path layer shifts to GND")
    axC.set_xticks(peaks)
    axC.set_ylim(0, max(gnd) * 1.25 + 1)
    axC.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_SUMMARY, dpi=150, bbox_inches="tight")
    print(f"\n[plot] Saved → {OUT_SUMMARY}")


def make_probability_figure(all_traces: dict) -> None:
    """3 rows (load levels) × 6 cols (AMF-ON, SMF-ON, UPF-ON, AMF-GND, SMF-GND, UPF-GND).
    Each panel: 3 instance probability traces over HO index."""
    n_rows = len(PROBS_LOADS)
    fig, axes = plt.subplots(n_rows, 6, figsize=(20, 3.0 * n_rows), sharex=True)
    fig.suptitle(f"Selection Probability vs HO Index  —  task='{FOCUS_TASK}'  "
                 f"(3 task-load multipliers, real Starlink TLE)",
                 fontsize=12, fontweight="bold")

    col_specs = [
        ("amf_on",  "AMF-ON",  "ISL"),
        ("smf_on",  "SMF-ON",  "ISL"),
        ("upf_on",  "UPF-ON",  "ISL"),
        ("amf_gnd", "AMF-GND", "GND"),
        ("smf_gnd", "SMF-GND", "GND"),
        ("upf_gnd", "UPF-GND", "GND"),
    ]

    for row, m in enumerate(PROBS_LOADS):
        traces = all_traces[m]
        n_ho = len(traces["amf_on"][0])
        ho_ids = np.arange(n_ho)
        for col, (key, layer_name, path_name) in enumerate(col_specs):
            ax = axes[row, col]
            for i in range(3):
                ax.plot(ho_ids, traces[key][i], lw=1.5,
                        color=INST_COLORS[i], linestyle=INST_STYLES[i],
                        label=f"{layer_name}-{i}")
            if row == 0:
                ax.set_title(f"{layer_name}  ({path_name})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"m = {m:.1f}\n\nprobability",
                              fontsize=10, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("HO index", fontsize=10)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PROBS, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {OUT_PROBS}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parameter experiments for the paper")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--rebuild-delays", action="store_true",
                    help="Force re-precompute of TLE-based delays")
    args = ap.parse_args()

    print("=" * 72)
    print("  Parameter Experiments  —  trend summary + per-load probability traces")
    print("=" * 72)

    delays = get_delays(N_HO, force_rebuild=args.rebuild_delays)
    delays_summary = (f"ISL={delays[:,0].mean():.1f}ms, "
                      f"GND={delays[:,1].mean():.1f}ms")

    sa = sweep_task_load(delays)
    sc = sweep_bg_peak(delays)
    traces = collect_traces_at_loads(delays)

    make_summary_figure(sa, sc, delays_summary)
    make_probability_figure(traces)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
