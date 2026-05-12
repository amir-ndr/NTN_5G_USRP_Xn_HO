#!/usr/bin/env python3
"""
plot_analysis.py — Extended scientific analysis for NTN dispatcher results.

Five analysis groups, each addressing a specific scientific claim:

  Group 1  Regime-shift response
           UPF instance weights across task-type switches.  The per-task
           scheduler separation is directly visible: weights jump when the
           task changes and resume from where they left off on return.

  Group 2  Load vs learned path split
           Scatter of bg_load_trgsat vs π_ISL colored by task type.
           Should show a phase transition near bg_ISL ≈ 0.65.

  Group 3  η_path sensitivity  (software simulation, no USRP needed)
           Cumulative global regret for η ∈ {0.05, 0.20, 0.50}.
           Too small → slow convergence; too large → oscillation.

  Group 4  Per-task oracle gap convergence
           Rolling-average (total_ms − global_oracle_ms) per task type
           vs per-task HO count.  Should decrease → empirical convergence.

  Group 5  Full-system instance-selection heatmap
           All 18 instance probabilities over HO index, with bg_load overlay.
           Coherent shift from ON to GND rows as TrgSAT load rises.

Usage:
    python3 plot_analysis.py [dispatch_log.csv]   # Groups 1,2,4,5 only
    python3 plot_analysis.py --eta-sweep          # adds Group 3 figure
    python3 plot_analysis.py --out results/       # output directory
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
LOG_CSV  = _HERE / "dispatch_log.csv"
OUT_A    = _HERE / "results_analysis_A.png"
OUT_B    = _HERE / "results_analysis_B_heatmap.png"
OUT_C    = _HERE / "results_analysis_C_eta_sweep.png"

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Shared constants ──────────────────────────────────────────────────────────

TASK_ORDER = ["gaming", "youtube", "browsing", "instagram", "mixed"]
TASK_COLORS = {
    "gaming":    "#1f77b4",
    "youtube":   "#ff7f0e",
    "browsing":  "#2ca02c",
    "instagram": "#d62728",
    "mixed":     "#9467bd",
}

# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fv(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - w + 1) : i + 1].mean()
    return out


def detect_switches(rows: list[dict]) -> list[tuple[int, str, str]]:
    """Return [(row_idx, from_task, to_task)] for each task-type change."""
    switches = []
    for i in range(1, len(rows)):
        a = rows[i - 1].get("task_type", "")
        b = rows[i].get("task_type", "")
        if a and b and a != b:
            switches.append((i, a, b))
    return switches


# ══════════════════════════════════════════════════════════════════════════════
#  Group 1 — Regime-shift response
# ══════════════════════════════════════════════════════════════════════════════

def _shade_task_bands(ax: plt.Axes, rows: list[dict], ho_ids: np.ndarray) -> None:
    """Fill background with light task-type color bands."""
    switches = detect_switches(rows)
    boundaries = [0] + [s[0] for s in switches] + [len(rows)]
    for seg_s, seg_e in zip(boundaries[:-1], boundaries[1:]):
        tt  = rows[seg_s].get("task_type", "mixed") if seg_s < len(rows) else "mixed"
        x0  = ho_ids[seg_s]
        x1  = ho_ids[min(seg_e, len(ho_ids) - 1)]
        ax.axvspan(x0, x1, alpha=0.07, color=TASK_COLORS.get(tt, "#888888"), zorder=0)


def _draw_switch_lines(ax: plt.Axes, rows: list[dict], ho_ids: np.ndarray) -> None:
    switches = detect_switches(rows)
    for idx, from_t, to_t in switches:
        x = ho_ids[min(idx, len(ho_ids) - 1)]
        ax.axvline(x=x, color="black", lw=0.9, linestyle="--", alpha=0.55, zorder=2)
        ax.text(x + 0.3, 0.97,
                f"{from_t[:3]}→{to_t[:3]}",
                transform=ax.get_xaxis_transform(),
                fontsize=6.0, va="top", rotation=90, color="#333333", alpha=0.8)


def plot_regime_shift(ax_on: plt.Axes, ax_gnd: plt.Axes, rows: list[dict]) -> None:
    ho_ids = np.array([fv(r, "ho_id") for r in rows])

    # ── Onboard UPF ───────────────────────────────────────────────────────────
    p0 = np.array([fv(r, "on_upf_p0") for r in rows])
    p1 = np.array([fv(r, "on_upf_p1") for r in rows])
    p2 = np.array([fv(r, "on_upf_p2") for r in rows])

    _shade_task_bands(ax_on, rows, ho_ids)
    ax_on.plot(ho_ids, p0, lw=1.8, color="#1f77b4", label="UPF-ON-0  (highest cap)")
    ax_on.plot(ho_ids, p1, lw=1.8, color="#ff7f0e", linestyle="--", label="UPF-ON-1")
    ax_on.plot(ho_ids, p2, lw=1.8, color="#2ca02c", linestyle=":",  label="UPF-ON-2  (unstable for insta/yt)")
    _draw_switch_lines(ax_on, rows, ho_ids)

    switches = detect_switches(rows)
    if not switches:
        ax_on.text(0.5, 0.5, "No task switches detected in this CSV.\n"
                   "Re-run controller.py (MIN_TASK_HOS=15 is now set).",
                   transform=ax_on.transAxes, ha="center", va="center",
                   fontsize=9, color="gray", style="italic")

    ax_on.set_ylabel("Selection probability")
    ax_on.set_title("Regime-Shift Response — Onboard UPF weights per task\n"
                    "(per-task schedulers preserve weights across task returns)")
    ax_on.legend(loc="upper right", fontsize=7.5)
    ax_on.set_ylim(-0.02, 1.08)
    ax_on.set_xticklabels([])
    ax_on.grid(True, alpha=0.28)

    # ── Ground UPF ────────────────────────────────────────────────────────────
    g0 = np.array([fv(r, "gnd_upf_p0") for r in rows])
    g1 = np.array([fv(r, "gnd_upf_p1") for r in rows])
    g2 = np.array([fv(r, "gnd_upf_p2") for r in rows])

    _shade_task_bands(ax_gnd, rows, ho_ids)
    ax_gnd.plot(ho_ids, g0, lw=1.8, color="#1f77b4", label="UPF-GND-0")
    ax_gnd.plot(ho_ids, g1, lw=1.8, color="#ff7f0e", linestyle="--", label="UPF-GND-1")
    ax_gnd.plot(ho_ids, g2, lw=1.8, color="#2ca02c", linestyle=":",  label="UPF-GND-2")
    _draw_switch_lines(ax_gnd, rows, ho_ids)

    ax_gnd.set_xlabel("Handover index")
    ax_gnd.set_ylabel("Selection probability")
    ax_gnd.set_title("Ground UPF weights per task  (all stable — less dramatic shifts)")
    ax_gnd.legend(loc="upper right", fontsize=7.5)
    ax_gnd.set_ylim(-0.02, 1.08)
    ax_gnd.grid(True, alpha=0.28)


# ══════════════════════════════════════════════════════════════════════════════
#  Group 2 — bg_load vs π_ISL scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_load_scatter(ax: plt.Axes, rows: list[dict]) -> None:
    for tt in TASK_ORDER:
        mask  = [r.get("task_type") == tt for r in rows]
        if not any(mask):
            continue
        bg = np.array([fv(r, "trgsat_bg") for r, m in zip(rows, mask) if m])
        pi = np.array([fv(r, "path_p_isl") for r, m in zip(rows, mask) if m])
        ax.scatter(bg, pi, s=20, alpha=0.65, color=TASK_COLORS[tt],
                   label=f"{tt} (n={int(m)} HOs)" if (m := sum(mask)) else tt,
                   zorder=3, edgecolors="none")

    ax.axvline(x=0.65, color="black", lw=1.3, linestyle="--", alpha=0.7,
               label="bg_ISL=0.65 crossover")
    ax.axhline(y=0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)

    # Add trend line (all tasks combined)
    bg_all = np.array([fv(r, "trgsat_bg") for r in rows])
    pi_all = np.array([fv(r, "path_p_isl") for r in rows])
    if len(bg_all) > 5:
        idx = np.argsort(bg_all)
        ax.plot(bg_all[idx], rolling_mean(pi_all[idx], 15),
                color="black", lw=1.5, alpha=0.5, label="trend (w=15)", zorder=5)

    ax.set_xlabel("bg_load TrgSAT (ISL node)")
    ax.set_ylabel("Learned π_ISL")
    ax.set_title("Load vs Learned Path Split\n"
                 "(phase transition expected at bg_ISL ≈ 0.65)")
    ax.legend(fontsize=7, markerscale=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.28)


# ══════════════════════════════════════════════════════════════════════════════
#  Group 4 — Per-task oracle gap convergence
# ══════════════════════════════════════════════════════════════════════════════

def plot_oracle_gap(ax: plt.Axes, rows: list[dict]) -> None:
    W = 12
    any_plotted = False
    for tt in TASK_ORDER:
        task_rows = [r for r in rows if r.get("task_type") == tt]
        if len(task_rows) < 4:
            continue
        gaps = np.array([max(0.0, fv(r, "total_ms") - fv(r, "global_oracle_ms"))
                         for r in task_rows])
        rm = rolling_mean(gaps, W)
        ax.plot(range(len(task_rows)), rm,
                color=TASK_COLORS[tt], lw=1.8,
                label=f"{tt} (n={len(task_rows)})")
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "Too few rows per task type.\nRe-run with balanced Markov chain.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray", style="italic")

    ax.axhline(y=0, color="black", lw=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Per-task HO count")
    ax.set_ylabel(f"Running avg (total − oracle)  ms  (w={W})")
    ax.set_title("Per-Task Oracle Gap Convergence\n"
                 "(decreasing gap → algorithm learning the oracle per task)")
    ax.legend(fontsize=7.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)


# ══════════════════════════════════════════════════════════════════════════════
#  Group 5 — Full-system instance-selection heatmap
# ══════════════════════════════════════════════════════════════════════════════

_INSTANCE_COLS = [
    # (csv_key, display_label)
    ("on_amf_p0",  "AMF-ON-0"),  ("on_amf_p1",  "AMF-ON-1"),  ("on_amf_p2",  "AMF-ON-2"),
    ("gnd_amf_p0", "AMF-GND-0"), ("gnd_amf_p1", "AMF-GND-1"), ("gnd_amf_p2", "AMF-GND-2"),
    ("on_smf_p0",  "SMF-ON-0"),  ("on_smf_p1",  "SMF-ON-1"),  ("on_smf_p2",  "SMF-ON-2"),
    ("gnd_smf_p0", "SMF-GND-0"), ("gnd_smf_p1", "SMF-GND-1"), ("gnd_smf_p2", "SMF-GND-2"),
    ("on_upf_p0",  "UPF-ON-0"),  ("on_upf_p1",  "UPF-ON-1"),  ("on_upf_p2",  "UPF-ON-2"),
    ("gnd_upf_p0", "UPF-GND-0"), ("gnd_upf_p1", "UPF-GND-1"), ("gnd_upf_p2", "UPF-GND-2"),
]
_LAYER_DIVIDERS = [2.5, 5.5, 8.5, 11.5, 14.5]   # between ON/GND groups and layers


def plot_heatmap(ax_bg: plt.Axes, ax_hm: plt.Axes, rows: list[dict]) -> None:
    ho_ids = np.array([fv(r, "ho_id") for r in rows])
    n_ho   = len(rows)
    n_inst = len(_INSTANCE_COLS)

    # Build probability matrix (n_inst × n_ho)
    matrix = np.array([[fv(r, col) for r in rows]
                       for col, _ in _INSTANCE_COLS])

    # ── bg_load mini panel ────────────────────────────────────────────────────
    trgsat_bg = np.array([fv(r, "trgsat_bg") for r in rows])
    tn_bg     = np.array([fv(r, "tn_bg")     for r in rows])
    ax_bg.fill_between(ho_ids, trgsat_bg, alpha=0.45, color="#1f77b4",
                       label="TrgSAT (ISL) bg_load")
    ax_bg.fill_between(ho_ids, tn_bg,     alpha=0.45, color="#ff7f0e",
                       label="TN (GND) bg_load")
    ax_bg.axhline(y=0.65, color="black", lw=0.9, linestyle="--", alpha=0.7,
                  label="crossover 0.65")
    ax_bg.set_ylim(0, 1.05)
    ax_bg.set_ylabel("bg_load")
    ax_bg.set_xlim(ho_ids[0], ho_ids[-1])
    ax_bg.set_xticklabels([])
    ax_bg.legend(loc="upper right", fontsize=7)
    ax_bg.grid(True, alpha=0.25)

    # Task-type color ticks along the top
    task_types = [r.get("task_type", "mixed") for r in rows]
    for i, (tt, x) in enumerate(zip(task_types, ho_ids)):
        ax_bg.plot(x, 0.02, "|", color=TASK_COLORS.get(tt, "#888"), ms=8, mew=1.5,
                   alpha=0.7, transform=ax_bg.get_xaxis_transform(), zorder=5)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    im = ax_hm.imshow(matrix, aspect="auto", cmap="YlOrRd",
                      vmin=0, vmax=1, origin="upper",
                      extent=[ho_ids[0], ho_ids[-1], n_inst - 0.5, -0.5])

    ax_hm.set_yticks(range(n_inst))
    ax_hm.set_yticklabels([lbl for _, lbl in _INSTANCE_COLS], fontsize=7.5)
    ax_hm.set_xlabel("Handover index")
    ax_hm.set_title("Full-System Instance-Selection Heatmap\n"
                    "(color = selection probability; observe coherent ON→GND shift as TrgSAT load rises)")

    # White dividers between layer groups
    for y in _LAYER_DIVIDERS:
        ax_hm.axhline(y=y, color="white", lw=2.0, zorder=3)

    # Layer labels on the right
    layer_labels = [
        (1.0,  "AMF-ON"),  (4.0,  "AMF-GND"),
        (7.0,  "SMF-ON"),  (10.0, "SMF-GND"),
        (13.0, "UPF-ON"),  (16.0, "UPF-GND"),
    ]
    for y_pos, lbl in layer_labels:
        ax_hm.text(ho_ids[-1] + 0.5, y_pos, lbl, va="center", fontsize=7,
                   color="#555555", clip_on=False)

    # Task switch lines on heatmap
    switches = detect_switches(rows)
    for idx, _, _ in switches:
        x = ho_ids[min(idx, n_ho - 1)]
        ax_hm.axvline(x=x, color="white", lw=1.0, linestyle="--", alpha=0.6, zorder=4)

    plt.colorbar(im, ax=ax_hm, label="Selection probability",
                 fraction=0.018, pad=0.01)


# ══════════════════════════════════════════════════════════════════════════════
#  Group 3 — η_path sensitivity  (pure software simulation)
# ══════════════════════════════════════════════════════════════════════════════

def run_eta_sweep(n_ho: int = 300, seed: int = 42) -> dict[float, np.ndarray]:
    """
    Run Dispatcher in software (no USRP) for three η_path values.
    Synthetic access costs derived from the same bg_load model.
    Returns {η: cum_global_regret array}.
    """
    import dispatcher as d

    ETA_PATH_VALUES = [0.05, 0.20, 0.50]
    rng = np.random.default_rng(seed)
    results: dict[float, np.ndarray] = {}

    # Markov transition (simplified uniform for the sweep — task mix doesn't change η story)
    _task_weights = [0.20, 0.20, 0.20, 0.20, 0.20]

    for eta in ETA_PATH_VALUES:
        # Patch module-level constants before instantiation
        d.ETA_PATH = eta
        d.ETA_X    = 0.05

        with tempfile.TemporaryDirectory() as tmp:
            disp         = d.Dispatcher(tmp)
            path_scheds  = {tt: d.PathScheduler() for tt in d.TASK_CYCLE}
            trgsat_node  = d.AccessNode("TrgSAT", ngap_ms=1.0, xn_base_ms=2.0)
            tn_node      = d.AccessNode("TN",     ngap_ms=0.5, xn_base_ms=5.0)

            task      = random.choice(d.TASK_CYCLE)
            task_ho   = 0
            cum_list: list[float] = []

            for ho_id in range(n_ho):
                trgsat_node.bg_load = d.trgsat_bg_load(ho_id)
                tn_node.bg_load     = d.tn_bg_load(ho_id)

                isl_ms = float(max(1.0, 4.0 + 0.8 * rng.standard_normal()))
                gnd_ms = float(max(1.0, 3.0 + 0.4 * rng.standard_normal()))

                cost_isl = trgsat_node.total_access_cost_ms(isl_ms)
                cost_gnd = tn_node.total_access_cost_ms(gnd_ms)

                sched = path_scheds[task]
                path  = sched.sample()
                sched.update(cost_isl, cost_gnd)

                _, row = disp.dispatch(
                    path=path, isl_ms=isl_ms, gnd_ms=gnd_ms,
                    task_type=task,
                    path_p_isl=sched.p_isl, path_p_gnd=sched.p_gnd,
                    access_cost_isl=cost_isl, access_cost_gnd=cost_gnd,
                    trgsat_bg=trgsat_node.bg_load, tn_bg=tn_node.bg_load,
                )
                cum_list.append(row["cum_global_regret_ms"])

                task_ho += 1
                if task_ho >= 15:
                    task    = random.choices(d.TASK_CYCLE, weights=_task_weights)[0]
                    task_ho = 0

            disp.close()
            results[eta] = np.array(cum_list)

    # Restore defaults
    d.ETA_PATH = 0.20
    d.ETA_X    = 0.05
    return results


def plot_eta_sweep(ax: plt.Axes, results: dict[float, np.ndarray]) -> None:
    palette = {0.05: "#d62728", 0.20: "#1f77b4", 0.50: "#2ca02c"}
    for eta, regrets in sorted(results.items()):
        ax.plot(regrets, lw=2.0, color=palette.get(eta, "black"),
                label=f"η_path = {eta}")

    ax.set_xlabel("Handover index (software simulation)")
    ax.set_ylabel("Cumulative global regret (ms)")
    ax.set_title("η_path Sensitivity — Cumulative Global Regret\n"
                 "(η=0.05: slow; η=0.50: oscillates; η=0.20: best trade-off)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.28)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="NTN dispatcher extended analysis plots")
    ap.add_argument("--eta-sweep", action="store_true",
                    help="Also run Group 3 η sensitivity simulation (~20 s)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if not LOG_CSV.exists():
        print(f"Error: {LOG_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(LOG_CSV)
    print(f"Loaded {len(rows)} rows from {LOG_CSV}")

    if not rows:
        print("dispatch_log.csv is empty — run controller.py first.")
        sys.exit(1)

    n_switches = len(detect_switches(rows))
    print(f"Task-type switches detected: {n_switches}")
    task_counts: dict[str, int] = {}
    for r in rows:
        tt = r.get("task_type", "unknown")
        task_counts[tt] = task_counts.get(tt, 0) + 1
    print("Task distribution:", {k: v for k, v in sorted(task_counts.items())})

    # ── Figure A: Groups 1, 2, 4 ──────────────────────────────────────────────
    figA = plt.figure(figsize=(18, 18))
    gsA  = figA.add_gridspec(3, 2, height_ratios=[3, 3, 3],
                              hspace=0.52, wspace=0.32)

    ax1_on  = figA.add_subplot(gsA[0, :])
    ax1_gnd = figA.add_subplot(gsA[1, :])
    plot_regime_shift(ax1_on, ax1_gnd, rows)

    ax2 = figA.add_subplot(gsA[2, 0])
    plot_load_scatter(ax2, rows)

    ax4 = figA.add_subplot(gsA[2, 1])
    plot_oracle_gap(ax4, rows)

    figA.suptitle("NTN Dispatcher — Analysis A  (Regime Shift · Load Scatter · Oracle Gap)",
                  fontsize=13, fontweight="bold")
    figA.savefig(OUT_A, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_A}")
    plt.close(figA)

    # ── Figure B: Group 5 heatmap ─────────────────────────────────────────────
    figB = plt.figure(figsize=(18, 14))
    gsB  = figB.add_gridspec(2, 1, height_ratios=[1.2, 5], hspace=0.06)

    ax_bg = figB.add_subplot(gsB[0])
    ax_hm = figB.add_subplot(gsB[1])
    plot_heatmap(ax_bg, ax_hm, rows)

    figB.suptitle("NTN Dispatcher — Analysis B  (Full-System Instance Heatmap)",
                  fontsize=13, fontweight="bold")
    figB.savefig(OUT_B, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_B}")
    plt.close(figB)

    # ── Figure C: Group 3 η sweep (optional) ──────────────────────────────────
    if args.eta_sweep:
        print("Running η-sweep simulation (300 HOs × 3 values) — may take ~20 s …")
        try:
            eta_results = run_eta_sweep(n_ho=300)
            figC, axC = plt.subplots(figsize=(11, 5))
            plot_eta_sweep(axC, eta_results)
            figC.suptitle("NTN Dispatcher — Analysis C  (η Sensitivity)",
                          fontsize=13, fontweight="bold")
            figC.tight_layout()
            figC.savefig(OUT_C, dpi=150, bbox_inches="tight")
            print(f"Saved {OUT_C}")
            plt.close(figC)
        except Exception as exc:
            print(f"η sweep failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
