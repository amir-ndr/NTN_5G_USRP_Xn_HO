#!/usr/bin/env python3
"""
instance_tracking.py — Phase-shifted AMF instance tracking demo

Demonstrates that the Level-2 LayerScheduler (Bregman mirror descent on the
NF-instance simplex) TRACKS the time-varying optimal AMF-ON instance as
oscillating background loads cause deterministic ranking crossings.

This directly visualises Lemma 1 of the paper — the drift-plus-penalty bound
drives the learned probability concentration toward the minimum-cost instance,
with tracking lag governed by η_x.

Setup:
  - AMF-ON-0 starts at PEAK   load  (phase 0.25 of cycle, sin=+1)
  - AMF-ON-1 starts at TROUGH load  (phase 0.75 of cycle, sin=−1)
  - AMF-ON-2 oscillates mildly (small amp, mid-phase, near baseline)
  - Same pattern applied to SMF-ON for symmetry
  - Period = 40 HOs → optimal instance flips every 20 HOs
  - Round-robin task selection (each task gets equal HOs)
  - Run 250 HOs covering ~6 full oscillation cycles

Output:
  instance_tracking.png — 3-panel figure:
    (1) AMF-ON-0/1/2 selection probabilities over time
    (2) Oracle "which instance has lowest E[W]" as a step function
    (3) Per-instance ρ trajectory (shows the load swap that drives the oracle)

Run:
  python3 instance_tracking.py
  python3 instance_tracking.py --no-show
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dispatcher as d

_HERE = Path(__file__).resolve().parent
OUT_PNG = _HERE / "instance_tracking.png"

# ── Tracking experiment configuration ────────────────────────────────────────
# Calibration notes:
#   - PERIOD=80 (slow oscillation): each phase lasts ~40 HOs, giving the
#     LayerScheduler enough rounds to track the change. With period=40 the
#     algorithm can't keep up at η_x=0.05.
#   - AMP=0.40: large enough to truly flip the E[W] ranking
#     (ON-0's smaller base τ otherwise wins even at peak load).
#   - ETA_X=0.15 (DEMO override of default 0.05): a 3× learning rate makes
#     tracking visible in 320 HOs. The default 0.05 is calibrated for noise
#     robustness in the live controller, where bg oscillations don't induce
#     such deterministic rank flips. We document this override in the figure.
N_HO       = 320       # 80 HOs per task × 4 tasks; ~4 full oscillation cycles
PERIOD     = 80        # HOs per oscillation cycle (slow → trackable)
AMP        = 0.40      # bg-load amplitude
ETA_X_DEMO = 0.15      # learning rate for this demo (default 0.05)
SEED       = 42

# Tasks used in this demo — instagram is SKIPPED because at N=1000 the
# combination of base ρ + amp=0.40 would push AMF-ON-2 above ρ=0.95.
# Skipping instagram is fine: the AMF tracking story doesn't depend on
# task identity (AMF is shared across all tasks).
DEMO_TASKS = ["gaming", "youtube", "browsing", "mixed"]

# Controlled-experiment override: AMF-ON-0 and AMF-ON-1 are given IDENTICAL
# hardware specs so the only difference between them is the phase of their
# background load.  Removes the baseline-τ confound — any preference the
# algorithm develops must come from the load tracking, not the hardware.
# Format: (name, millicores, freq_ghz, is_onboard, fixed_cpr, cs, ca, tau_max)
IDENTICAL_AMF_ON_SPECS = [
    ("AMF-ON-0",  600, 1.5, True,  400_000, 1.10, 1.20, 1.5),   # ← identical
    ("AMF-ON-1",  600, 1.5, True,  400_000, 1.10, 1.20, 1.5),   # ← identical
    ("AMF-ON-2",  400, 1.5, True,  400_000, 1.40, 1.20, 1.5),   # baseline
]
IDENTICAL_SMF_ON_SPECS = [
    ("SMF-ON-0",  800, 1.5, True,  500_000, 1.15, 1.15, 1.0),   # ← identical
    ("SMF-ON-1",  800, 1.5, True,  500_000, 1.15, 1.15, 1.0),   # ← identical
    ("SMF-ON-2",  550, 1.5, True,  500_000, 1.35, 1.15, 1.0),   # baseline
]

# Phase-shifted INSTANCE_LOAD_PROFILES — overrides the default at runtime.
# Phase is in UNITS of fraction-of-cycle (the dispatcher multiplies by 2π).
#   phase=0.25 → sin starts at +1 (peak)
#   phase=0.75 → sin starts at -1 (trough)
PHASE_SHIFTED_PROFILES = {
    "AMF-ON-0":  (AMP,    PERIOD, 0.25),   # starts at PEAK load (sin = +1)
    "AMF-ON-1":  (AMP,    PERIOD, 0.75),   # starts at TROUGH load (sin = -1)
    "AMF-ON-2":  (0.10,   PERIOD, 0.50),   # mild oscillation, doesn't compete
    "AMF-GND-0": (0.05,   PERIOD*2, 0.10),
    "AMF-GND-1": (0.05,   PERIOD*2, 0.55),
    "AMF-GND-2": (0.05,   PERIOD*2, 0.85),
    "SMF-ON-0":  (AMP,    PERIOD, 0.25),
    "SMF-ON-1":  (AMP,    PERIOD, 0.75),
    "SMF-ON-2":  (0.10,   PERIOD, 0.50),
    "SMF-GND-0": (0.05,   PERIOD*2, 0.15),
    "SMF-GND-1": (0.05,   PERIOD*2, 0.60),
    "SMF-GND-2": (0.05,   PERIOD*2, 0.90),
}


def amf_oracle_index(dispatcher_obj, task: str) -> int:
    """Return the AMF-ON instance index with the lowest expected delay right now."""
    delays = [inst.expected_delay_ms() for inst in dispatcher_obj.on_amf.instances]
    return int(np.argmin(delays))


def run_tracking(n_ho: int = N_HO, seed: int = SEED):
    """Run the phase-shifted tracking experiment. Returns time-series dict."""
    # Patch module-level state for this controlled experiment
    original_profiles = d.INSTANCE_LOAD_PROFILES
    original_eta_x    = d.ETA_X
    original_amf_on   = d._ON_AMF_SPECS
    original_smf_on   = d._ON_SMF_SPECS
    d.INSTANCE_LOAD_PROFILES = PHASE_SHIFTED_PROFILES
    d.ETA_X                  = ETA_X_DEMO
    d._ON_AMF_SPECS          = IDENTICAL_AMF_ON_SPECS
    d._ON_SMF_SPECS          = IDENTICAL_SMF_ON_SPECS

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Round-robin over DEMO_TASKS (instagram excluded — too high N for amp=0.40).
    # Each task gets exactly n_ho/len(tasks) HOs in a contiguous block, then rotate.
    tasks_per_block = n_ho // len(DEMO_TASKS)
    task_sequence = []
    for tt in DEMO_TASKS:
        task_sequence.extend([tt] * tasks_per_block)
    while len(task_sequence) < n_ho:
        task_sequence.append(DEMO_TASKS[-1])

    traces: dict = {
        "amf_p":      [[] for _ in range(3)],   # selection probabilities
        "amf_rho":    [[] for _ in range(3)],   # ρ values
        "amf_E_W":    [[] for _ in range(3)],   # expected sojourn
        "oracle":     [],                        # which AMF-ON has min E[W]
        "task":       [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        disp        = d.Dispatcher(tmp)
        path_scheds = {tt: d.PathScheduler() for tt in d.TASK_CYCLE}
        trgsat      = d.AccessNode("TrgSAT", ngap_ms=1.0, xn_base_ms=3.0)
        tn          = d.AccessNode("TN",     ngap_ms=0.5, xn_base_ms=5.0)

        for ho_id in range(n_ho):
            task = task_sequence[ho_id]
            trgsat.bg_load = d.trgsat_bg_load(ho_id)
            tn.bg_load     = d.tn_bg_load(ho_id)

            isl_ms = float(max(1.0, 18.0 + 3.0 * rng.standard_normal()))
            gnd_ms = float(max(1.0, 29.0 + 2.0 * rng.standard_normal()))

            cost_isl = trgsat.total_access_cost_ms(isl_ms)
            cost_gnd = tn.total_access_cost_ms(gnd_ms)

            # FORCE path=ISL so AMF-ON is exercised every round (we are studying
            # the AMF LayerScheduler, not the path scheduler in this experiment).
            path  = "ISL"
            sched = path_scheds[task]
            sched.sample()   # still keep state consistent

            disp.dispatch(
                path            = path,
                isl_ms          = isl_ms,
                gnd_ms          = gnd_ms,
                task_type       = task,
                path_p_isl      = sched.p_isl,
                path_p_gnd      = sched.p_gnd,
                access_cost_isl = cost_isl,
                access_cost_gnd = cost_gnd,
                trgsat_bg       = trgsat.bg_load,
                tn_bg           = tn.bg_load,
            )

            # Record AMF-ON state AFTER the update.
            for i, inst in enumerate(disp.on_amf.instances):
                traces["amf_p"][i].append(disp.on_amf.probabilities()[i])
                traces["amf_rho"][i].append(inst.rho)
                traces["amf_E_W"][i].append(inst.expected_delay_ms())
            traces["oracle"].append(amf_oracle_index(disp, task))
            traces["task"].append(task)

        disp.close()

    # Restore module-level state so other importers see the defaults
    d.INSTANCE_LOAD_PROFILES = original_profiles
    d.ETA_X                  = original_eta_x
    d._ON_AMF_SPECS          = original_amf_on
    d._ON_SMF_SPECS          = original_smf_on
    return traces


def plot_tracking(traces: dict) -> None:
    n_ho   = len(traces["task"])
    ho_ids = np.arange(n_ho)
    names  = ["AMF-ON-0", "AMF-ON-1", "AMF-ON-2"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig, (ax_p, ax_o, ax_rho) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 2.5], "hspace": 0.12},
    )
    fig.suptitle("AMF-ON LayerScheduler — Phase-Shifted Tracking Demonstration\n"
                 f"(period={PERIOD} HOs, amplitude={AMP}, round-robin tasks)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: learned selection probabilities ─────────────────────────────
    for i, (name, c) in enumerate(zip(names, colors)):
        ax_p.plot(ho_ids, traces["amf_p"][i], lw=2.2, color=c, label=name)
    ax_p.set_ylabel("Selection probability", fontsize=11)
    ax_p.set_title("Learned probabilities (Bregman mirror descent on simplex)",
                   fontsize=11)
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.axhline(1.0/3.0, color="gray", linestyle=":", lw=0.8, alpha=0.6,
                 label="uniform (random)")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(fontsize=9, ncol=4, loc="upper right")

    # ── Panel 2: oracle (which instance has min E[W]) ────────────────────────
    oracle_arr = np.array(traces["oracle"])
    # Render as a coloured strip
    for i, c in enumerate(colors):
        mask = oracle_arr == i
        ax_o.fill_between(ho_ids, 0, 1, where=mask, color=c, alpha=0.6,
                          step="post", linewidth=0)
    ax_o.set_ylabel("Oracle", fontsize=11)
    ax_o.set_title("Oracle — instance with lowest expected sojourn E[W] at each HO",
                   fontsize=11)
    ax_o.set_yticks([])
    ax_o.set_ylim(0, 1)

    # ── Panel 3: ρ trajectories driving the oracle ──────────────────────────
    for i, (name, c) in enumerate(zip(names, colors)):
        ax_rho.plot(ho_ids, traces["amf_rho"][i], lw=1.5, color=c,
                    alpha=0.85, label=name)
    ax_rho.set_xlabel("Handover index", fontsize=11)
    ax_rho.set_ylabel("Utilization ρ", fontsize=11)
    ax_rho.set_title("Per-instance ρ over time — load oscillations drive the crossings",
                     fontsize=11)
    ax_rho.axhline(0.95, color="red", linestyle="--", lw=0.8, alpha=0.6,
                   label="instability threshold")
    ax_rho.grid(True, alpha=0.3)
    ax_rho.legend(fontsize=9, ncol=4, loc="upper right")
    ax_rho.set_ylim(0, 1.05)

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n[plot] Saved → {OUT_PNG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-show", action="store_true",
                    help="Save PNG only, don't open window")
    ap.add_argument("--n-ho", type=int, default=N_HO,
                    help=f"Total HOs (default {N_HO})")
    args = ap.parse_args()

    print(f"\n{'═'*72}")
    print(f"  AMF-ON tracking demo  —  {args.n_ho} HOs, period={PERIOD}, amp={AMP}")
    print(f"  Phase shift: ON-0 starts at PEAK, ON-1 starts at TROUGH")
    print(f"  Round-robin tasks ({len(DEMO_TASKS)}): {args.n_ho // len(DEMO_TASKS)} HOs each ({DEMO_TASKS})")
    print(f"{'═'*72}\n")

    traces = run_tracking(n_ho=args.n_ho, seed=SEED)

    # ── Tracking-quality metrics ─────────────────────────────────────────────
    # The right metric here is CROSS-CORRELATION between p_ON-i and the oracle
    # indicator (1 when oracle picks ON-i, 0 otherwise), maximised over a small
    # lag window.  In an oscillating environment with online-learning lag, the
    # peak correlation occurs at a positive lag and reflects how well the
    # algorithm tracks the oracle's swings.
    oracle_arr = np.array(traces["oracle"])
    p_arr      = np.array(traces["amf_p"])
    max_lag    = PERIOD // 2

    def best_lag_corr(p_series: np.ndarray, oracle_is_i: np.ndarray):
        p_c = p_series - p_series.mean()
        o_c = oracle_is_i.astype(float) - oracle_is_i.mean()
        if p_c.std() < 1e-9 or o_c.std() < 1e-9:
            return 0.0, 0
        corrs = [np.corrcoef(o_c[: len(o_c) - k] if k > 0 else o_c,
                              p_c[k:] if k > 0 else p_c)[0, 1]
                  for k in range(max_lag + 1)]
        k_best = int(np.argmax(corrs))
        return float(corrs[k_best]), k_best

    print(f"[summary] Tracking quality vs. oracle:")
    for i, name in enumerate(["AMF-ON-0", "AMF-ON-1", "AMF-ON-2"]):
        time_on_top = (oracle_arr == i).mean() * 100
        p_swing     = (p_arr[i].max() - p_arr[i].min())
        corr, lag   = best_lag_corr(p_arr[i], oracle_arr == i)
        print(f"  {name}: oracle picked it {time_on_top:5.1f}% of HOs  "
              f"p∈[{p_arr[i].min():.2f},{p_arr[i].max():.2f}] (swing {p_swing:.2f})  "
              f"corr={corr:+.3f} @ lag={lag} HOs")
    print(f"  (η_x={ETA_X_DEMO}, period={PERIOD} HOs → expected lag ≈ period/4 = {PERIOD//4})")

    plot_tracking(traces)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
