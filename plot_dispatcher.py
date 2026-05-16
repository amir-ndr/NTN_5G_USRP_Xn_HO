#!/usr/bin/env python3
"""
plot_dispatcher.py — Visualise dispatcher learning results

10-panel figure (5×2 grid):
  1.   PathScheduler — learned π_ISL / π_GND
  2.   Access cost signals — ISL vs GND  (driving signal for Level-1)
  A-L. Cumulative regret — log-log with fitted slope  (proves theorem)
  A-R. Regret / √T — flat or decreasing = O(√T) confirmed
  3.   End-to-end delay — rolling mean ± 1σ band  (convergence trend)
  6.   Cumulative regret decomposition — access + NF instance + global
  5.   Per-task mean latency — Bregman vs Random stacked bars
  7.   ON-UPF instance weights — key non-stationarity response
  B-L. ON  B_mult convergence — exploitation arm on onboard NFs
  B-R. GND B_mult convergence — exploitation arm on ground NFs

Separate debug figure (debug_instagram.png):
  UPF selection + delay during instagram blocks — verify instability learning

Usage:
    python3 plot_dispatcher.py
    python3 plot_dispatcher.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_HERE     = Path(__file__).resolve().parent
LOG_CSV   = _HERE / "results/default_ls1/dispatch_log_default_ls1.csv"
RAND_CSV  = _HERE / "results/default_ls1/random_log_default_ls1.csv"
OUT_PNG    = _HERE / "results/default_ls1/results_dispatcher_ls1.png"
DEBUG_PNG  = _HERE / "results/default_ls1/debug_instagram.png"
LEVEL1_PNG = _HERE / "results/default_ls1/level1_dynamics_ls1.png"

CLIP_MS        = 70.0    # y-axis display clip for delay panels
OUTLIER_THRESH = 150.0   # exclude from per-task latency averages

# B_mult bounds (Level-2; must match dispatcher.py)
B_MULT_MIN = 0.7
B_MULT_MAX = 1.5

# Level-1 PathScheduler bounds (must match dispatcher.py)
B_PATH_MAX   = 0.20
RHO_MIN_PATH = 0.02
TAU_ISL_MS   = 4.0   # trgSAT xn_base_ms
TAU_GND_MS   = 5.0   # TN     xn_base_ms

C_ISL   = "#2196F3"
C_GND   = "#FF5722"
C_BEST  = "#4CAF50"
C_RAND  = "#9E9E9E"
C_RAND2 = "#757575"

TASK_ORDER = ["gaming", "youtube", "browsing", "instagram", "mixed"]
TASK_COLORS = {
    "gaming":    "#BBDEFB",
    "youtube":   "#FFCDD2",
    "browsing":  "#C8E6C9",
    "instagram": "#FFE0B2",
    "mixed":     "#E1BEE7",
}
TASK_BAR_COLORS = {
    "gaming":    "#1E88E5",
    "youtube":   "#E53935",
    "browsing":  "#43A047",
    "instagram": "#FB8C00",
    "mixed":     "#8E24AA",
}
TASK_LINE_COLORS = {
    "gaming":    "#1E88E5",
    "youtube":   "#E53935",
    "browsing":  "#43A047",
    "instagram": "#FB8C00",
    "mixed":     "#8E24AA",
}
TASK_SHORT = {
    "gaming": "GAM", "youtube": "YT", "browsing": "BR",
    "instagram": "IG", "mixed": "MX",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_log(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            print(f"[plot] ERROR: {path} not found. Run the controller first.")
            sys.exit(1)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fv(row: dict, key: str) -> float:
    return float(row[key])


def add_task_shading(ax, ho_ids: np.ndarray, task_types: list[str]) -> None:
    """Semi-transparent task-type bands + vertical boundary lines."""
    if not task_types:
        return
    blocks: list[tuple[float, float, str]] = []
    prev_tt = task_types[0]
    start_x = ho_ids[0]
    for i in range(1, len(ho_ids)):
        if task_types[i] != prev_tt:
            blocks.append((start_x, ho_ids[i - 1], prev_tt))
            ax.axvline(float(ho_ids[i]) - 0.5,
                       color="#555555", ls=":", lw=0.7, alpha=0.5, zorder=1)
            start_x = ho_ids[i]
            prev_tt = task_types[i]
    blocks.append((start_x, ho_ids[-1], prev_tt))
    for x0, x1, tt in blocks:
        ax.axvspan(float(x0) - 0.5, float(x1) + 0.5,
                   alpha=0.12, color=TASK_COLORS.get(tt, "#F5F5F5"), zorder=0)
        mid = (float(x0) + float(x1)) / 2
        ax.text(mid, 1.0, TASK_SHORT.get(tt, tt[:3].upper()),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.5,
                color="#444444", alpha=0.8)


def rolling_mean_valid(
    values: np.ndarray, clip_ms: float, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean over non-clipped points only."""
    n = len(values)
    out_x: list[int]   = []
    out_y: list[float] = []
    for i in range(n):
        lo    = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        valid = chunk[chunk < clip_ms]
        if len(valid) > 0:
            out_x.append(i)
            out_y.append(float(valid.mean()))
    return np.array(out_x, dtype=int), np.array(out_y)


def rolling_nan_stats(
    values: np.ndarray, clip: float, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling mean and ±1σ, ignoring points above clip.
    Returns (x_indices, means, stds) where x_indices are positions in values."""
    n = len(values)
    out_x, out_mean, out_std = [], [], []
    for i in range(n):
        lo    = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        valid = chunk[chunk < clip]
        if len(valid) > 0:
            out_x.append(i)
            out_mean.append(float(valid.mean()))
            out_std.append(float(valid.std()) if len(valid) > 1 else 0.0)
    return (np.array(out_x, dtype=int),
            np.array(out_mean),
            np.array(out_std))


# ══════════════════════════════════════════════════════════════════════════════
#  Debug figure — instagram instability response (separate PNG)
# ══════════════════════════════════════════════════════════════════════════════

def save_debug_instagram(
    rows:       list[dict],
    ho_ids:     np.ndarray,
    task_types: list[str],
) -> None:
    instagram_mask = np.array([t == "instagram" for t in task_types])
    if not instagram_mask.any():
        print("[debug] No instagram HOs found — skipping debug_instagram.png")
        return

    on_upf_p0 = np.array([fv(r, "on_upf_p0") for r in rows])
    on_upf_p1 = np.array([fv(r, "on_upf_p1") for r in rows])
    on_upf_p2 = np.array([fv(r, "on_upf_p2") for r in rows])
    upf_ms    = np.array([fv(r, "upf_ms")    for r in rows])
    upf_rho   = np.array([fv(r, "upf_rho")   for r in rows])

    ig_ho   = ho_ids[instagram_mask]
    ig_p2   = on_upf_p2[instagram_mask]
    ig_p0   = on_upf_p0[instagram_mask]
    ig_p1   = on_upf_p1[instagram_mask]
    ig_upf  = upf_ms[instagram_mask]
    ig_rho  = upf_rho[instagram_mask]

    # Find instagram blocks (groups of contiguous instagram HOs)
    switches = [(i, task_types[i - 1], task_types[i])
                for i in range(1, len(task_types)) if task_types[i] != task_types[i - 1]]
    ig_entry_ho = [ho_ids[i] for i, _, nxt in switches if nxt == "instagram"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        "DEBUG — Instagram UPF Instability Response\n"
        "(ON-UPF-0/1 unstable at N=1000 × CPR / capacity_hz > 0.95 → algorithm must select ON-UPF-2)",
        fontsize=10, fontweight="bold"
    )

    # ── Panel 1: ON-UPF selection probabilities during instagram HOs ──────────
    ax1 = axes[0]
    ax1.scatter(ig_ho, ig_p2, color="#43A047", s=25, alpha=0.80, zorder=4,
                label="P(select ON-UPF-2)  — only stable NTN UPF @ instagram")
    ax1.scatter(ig_ho, ig_p0, color="#E53935", s=18, alpha=0.60, zorder=3,
                label="P(select ON-UPF-0)  — unstable @ instagram")
    ax1.scatter(ig_ho, ig_p1, color="#FFA726", s=18, alpha=0.60, zorder=3,
                label="P(select ON-UPF-1)  — unstable @ instagram")

    if len(ig_p2) > 4:
        win = max(3, len(ig_p2) // 5)
        rm  = np.convolve(ig_p2, np.ones(win) / win, mode="valid")
        ax1.plot(ig_ho[win - 1:], rm, color="#2E7D32", lw=2.2,
                 label=f"P(ON-UPF-2) rolling mean (w={win})")

    ax1.axhline(1 / 3, color="#888888", ls=":", lw=0.9, label="Uniform prior (1/3)")
    ax1.axhline(1.0,   color="#43A047", ls="--", lw=0.8, alpha=0.5,
                label="Target: P=1.0 (only stable instance)")

    for hid in ig_entry_ho:
        ax1.axvline(hid, color="#FB8C00", ls="--", lw=1.3, alpha=0.7)

    ax1.set_title(
        "ON-UPF Instance Selection Probability — Instagram HOs Only\n"
        "Algorithm should converge P(ON-UPF-2) → 1.0 within each instagram block",
        fontsize=9
    )
    ax1.set_ylabel("Selection probability")
    ax1.set_ylim(-0.02, 1.12)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: UPF delay and ρ during instagram HOs ─────────────────────────
    ax2 = axes[1]
    ig_upf_clipped = np.minimum(ig_upf, 100.0)
    is_unstable    = ig_upf > 500

    ax2.scatter(ig_ho[~is_unstable], ig_upf_clipped[~is_unstable],
                color=C_ISL, s=20, alpha=0.75, zorder=4,
                label="UPF delay (ms, clipped at 100)")
    if is_unstable.any():
        ax2.scatter(ig_ho[is_unstable], np.full(is_unstable.sum(), 98.0),
                    marker="^", color="#E53935", s=50, zorder=5,
                    label=f"Unstable instance selected ({is_unstable.sum()} HOs, delay > 500 ms)")

    # Secondary axis: ρ
    ax2r = ax2.twinx()
    ax2r.scatter(ig_ho, ig_rho, color="#9C27B0", s=14, alpha=0.55, zorder=3,
                 marker="D")
    ax2r.axhline(0.95, color="#9C27B0", ls="--", lw=0.9, alpha=0.7,
                 label="ρ = 0.95 (instability threshold)")
    ax2r.set_ylabel("UPF utilisation ρ", color="#9C27B0", fontsize=9)
    ax2r.tick_params(axis="y", labelcolor="#9C27B0")
    ax2r.set_ylim(0, 1.15)

    if len(ig_upf_clipped) > 4:
        win = max(3, len(ig_upf_clipped) // 5)
        rm_upf = np.convolve(ig_upf_clipped, np.ones(win) / win, mode="valid")
        ax2.plot(ig_ho[win - 1:], rm_upf, color=C_ISL, lw=2.0,
                 label=f"UPF delay rolling mean (w={win})")

    for hid in ig_entry_ho:
        ax2.axvline(hid, color="#FB8C00", ls="--", lw=1.3, alpha=0.7)

    ax2.set_title(
        "UPF Delay and Utilisation ρ — Instagram HOs Only\n"
        "Delay should decrease as algorithm shifts to ON-UPF-2",
        fontsize=9
    )
    ax2.set_xlabel("Handover index")
    ax2.set_ylabel("UPF sojourn delay (ms, clipped at 100)")
    ax2.set_ylim(0, 105)

    # Combine legends from ax2 + ax2r
    h2, l2   = ax2.get_legend_handles_labels()
    h2r, l2r = ax2r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(DEBUG_PNG, dpi=150, bbox_inches="tight")
    print(f"[debug] Saved → {DEBUG_PNG}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Level-1 PathScheduler dynamics figure (paper Algorithm 1 visualisation)
# ══════════════════════════════════════════════════════════════════════════════

def save_level1_dynamics(
    rows: list[dict], ho_ids: np.ndarray, task_types: list[str]
) -> None:
    """Visualise the paper's dynamic projection bound on B_i (Algorithm 1, Step 1b):
        B_i[t+1] ∈ [0, min{B_max, Q_i[t] + γ_i[t]}]
    Panels:
      1. B_ISL, B_GND  with B_PATH_MAX and per-path B_floor reference lines.
      2. Q_ISL, Q_GND  (paper Eq. 1 backlog evolution).
      3. Dynamic projection cap min{B_PATH_MAX, Q+γ} vs actual B per path —
         shows which constraint binds round-by-round.
      4. π_ISL trajectory with chosen-path scatter — does Level-1 learn anything
         beyond geometry?
    """
    required = ("B_isl", "B_gnd", "Q_isl", "Q_gnd", "gamma_isl", "gamma_gnd")
    if not rows or any(col not in rows[0] for col in required):
        missing = [c for c in required if not rows or c not in rows[0]]
        print(f"[level1] dispatch_log missing columns {missing} — re-run "
              f"controller.py to populate fresh CSVs.")
        return

    B_isl = np.array([fv(r, "B_isl")     for r in rows])
    B_gnd = np.array([fv(r, "B_gnd")     for r in rows])
    Q_isl = np.array([fv(r, "Q_isl")     for r in rows])
    Q_gnd = np.array([fv(r, "Q_gnd")     for r in rows])
    g_isl = np.array([fv(r, "gamma_isl") for r in rows])
    g_gnd = np.array([fv(r, "gamma_gnd") for r in rows])
    p_isl = np.array([fv(r, "path_p_isl") for r in rows])
    paths = [r.get("path", "") for r in rows]

    B_floor_isl = RHO_MIN_PATH / TAU_ISL_MS
    B_floor_gnd = RHO_MIN_PATH / TAU_GND_MS
    cap_isl     = np.minimum(B_PATH_MAX, Q_isl + g_isl)
    cap_gnd     = np.minimum(B_PATH_MAX, Q_gnd + g_gnd)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Level-1 PathScheduler Dynamics  (Algorithm 1, Step 1b)",
                 fontsize=14, fontweight="bold", y=0.995)

    # ── 1. B trajectories ────────────────────────────────────────────────────
    ax = axes[0]
    add_task_shading(ax, ho_ids, task_types)
    ax.plot(ho_ids, B_isl, color=C_ISL, lw=1.4, label=r"$B_{ISL}$")
    ax.plot(ho_ids, B_gnd, color=C_GND, lw=1.4, label=r"$B_{GND}$")
    ax.axhline(B_PATH_MAX,  color="black",   ls="--", lw=0.8, alpha=0.6,
               label=f"B_PATH_MAX ({B_PATH_MAX})")
    ax.axhline(B_floor_isl, color=C_ISL,     ls=":",  lw=0.8, alpha=0.6,
               label=f"B_floor ISL ({B_floor_isl:.3f})")
    ax.axhline(B_floor_gnd, color=C_GND,     ls=":",  lw=0.8, alpha=0.6,
               label=f"B_floor GND ({B_floor_gnd:.3f})")
    ax.set_ylabel(r"$B_i$  (1/ms)")
    ax.set_title("Service-rate estimate per path")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 2. Q trajectories ────────────────────────────────────────────────────
    ax = axes[1]
    add_task_shading(ax, ho_ids, task_types)
    ax.plot(ho_ids, Q_isl, color=C_ISL, lw=1.4, label=r"$Q_{ISL}$")
    ax.plot(ho_ids, Q_gnd, color=C_GND, lw=1.4, label=r"$Q_{GND}$")
    ax.set_ylabel(r"$Q_i$  (1/ms)")
    ax.set_title("Dispatcher backlog  (paper Eq. 1)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 3. Dynamic cap vs actual B (per path) ────────────────────────────────
    ax = axes[2]
    add_task_shading(ax, ho_ids, task_types)
    ax.plot(ho_ids, cap_isl, color=C_ISL, ls="--", lw=1.0, alpha=0.7,
            label=r"cap$_{ISL}$ = min{B_max, Q+γ}")
    ax.plot(ho_ids, B_isl,   color=C_ISL, lw=1.4,
            label=r"$B_{ISL}$ (actual)")
    ax.plot(ho_ids, cap_gnd, color=C_GND, ls="--", lw=1.0, alpha=0.7,
            label=r"cap$_{GND}$ = min{B_max, Q+γ}")
    ax.plot(ho_ids, B_gnd,   color=C_GND, lw=1.4,
            label=r"$B_{GND}$ (actual)")
    ax.axhline(B_PATH_MAX, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.set_ylabel(r"Service rate (1/ms)")
    ax.set_title("Dynamic projection cap vs actual B  "
                 "(when cap line meets actual line, that bound binds)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 4. π_ISL with chosen-path scatter ────────────────────────────────────
    ax = axes[3]
    add_task_shading(ax, ho_ids, task_types)
    ax.plot(ho_ids, p_isl, color="#7B1FA2", lw=1.6, label=r"$\pi_{ISL}$")
    ax.axhline(0.5, color="black", ls=":", lw=0.8, alpha=0.5)
    chose_isl = np.array([p == "ISL" for p in paths])
    if chose_isl.any():
        ax.scatter(ho_ids[chose_isl],  np.full(chose_isl.sum(), 1.03),
                   color=C_ISL, marker="|", s=30, label="ISL chosen")
    if (~chose_isl).any():
        ax.scatter(ho_ids[~chose_isl], np.full((~chose_isl).sum(), -0.03),
                   color=C_GND, marker="|", s=30, label="GND chosen")
    ax.set_xlabel("Handover index")
    ax.set_ylabel(r"$\pi_{ISL}$")
    ax.set_ylim(-0.08, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Path selection probability and realised choices")
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(LEVEL1_PNG, dpi=150, bbox_inches="tight")
    print(f"[level1] Saved → {LEVEL1_PNG}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main figure
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    rows = load_log(LOG_CSV, required=True)
    if not rows:
        print("[plot] dispatch_log is empty.")
        sys.exit(1)

    rand_rows = load_log(RAND_CSV, required=False)
    has_rand  = len(rand_rows) > 0
    if not has_rand:
        print("[plot] random_log.csv not found — random baseline overlay skipped.")

    # ── Bregman data ──────────────────────────────────────────────────────────
    n          = len(rows)
    ho_ids     = np.array([fv(r, "ho_id")              for r in rows])
    prop_ms    = np.array([fv(r, "prop_ms")            for r in rows])
    amf_ms     = np.array([fv(r, "amf_ms")             for r in rows])
    smf_ms     = np.array([fv(r, "smf_ms")             for r in rows])
    upf_ms     = np.array([fv(r, "upf_ms")             for r in rows])
    total_ms   = np.array([fv(r, "total_ms")           for r in rows])
    oracle     = np.array([fv(r, "global_oracle_ms")   for r in rows])

    path_p_isl   = np.array([fv(r, "path_p_isl")          for r in rows])
    path_p_gnd   = np.array([fv(r, "path_p_gnd")          for r in rows])
    acc_cost_isl = np.array([fv(r, "access_cost_isl")      for r in rows])
    acc_cost_gnd = np.array([fv(r, "access_cost_gnd")      for r in rows])

    cum_access_reg = np.array([fv(r, "cum_access_regret_ms") for r in rows])
    cum_inst_reg   = np.array([fv(r, "cum_inst_regret_ms")   for r in rows])
    cum_global_reg = np.array([fv(r, "cum_global_regret_ms") for r in rows])
    # global_reg_ms not used after panel 4 was removed

    isl_mask   = np.array([r["path"] == "ISL" for r in rows])
    gnd_mask   = ~isl_mask
    task_types = [r["task_type"] for r in rows]

    # ON-UPF instance probabilities (only UPF shown in its own panel)
    on_upf_p = (
        np.array([fv(r, "on_upf_p0") for r in rows]),
        np.array([fv(r, "on_upf_p1") for r in rows]),
        np.array([fv(r, "on_upf_p2") for r in rows]),
    )

    # B_mult per instance (exploitation arm state)
    has_bmult = "on_amf_B0" in rows[0]
    if has_bmult:
        def bmult_layer(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.array([fv(r, f"{prefix}_B0") for r in rows]),
                np.array([fv(r, f"{prefix}_B1") for r in rows]),
                np.array([fv(r, f"{prefix}_B2") for r in rows]),
            )
        on_amf_B  = bmult_layer("on_amf")
        on_smf_B  = bmult_layer("on_smf")
        on_upf_B  = bmult_layer("on_upf")
        gnd_amf_B = bmult_layer("gnd_amf")
        gnd_smf_B = bmult_layer("gnd_smf")
        gnd_upf_B = bmult_layer("gnd_upf")

    # ── Random baseline ───────────────────────────────────────────────────────
    rand_ho_ids         = np.array([])
    rand_total          = np.array([])
    rand_cum_reg        = np.array([])
    rand_cum_access_reg = np.array([])
    rand_cum_inst_reg   = np.array([])
    rand_task_types: list[str] = []
    rand_prop  = rand_amf = rand_smf = rand_upf = np.array([])
    nr = 0

    if has_rand:
        nr                  = len(rand_rows)
        rand_ho_ids         = np.array([fv(r, "ho_id")                  for r in rand_rows])
        rand_total          = np.array([fv(r, "total_ms")               for r in rand_rows])
        rand_cum_reg        = np.array([fv(r, "cum_global_regret_ms")   for r in rand_rows])
        rand_cum_access_reg = np.array([fv(r, "cum_access_regret_ms")   for r in rand_rows])
        rand_cum_inst_reg   = np.array([fv(r, "cum_inst_regret_ms")     for r in rand_rows])
        rand_isl_mask       = np.array([r["path"] == "ISL"              for r in rand_rows])
        rand_gnd_mask       = ~rand_isl_mask
        rand_task_types     = [r["task_type"]                           for r in rand_rows]
        rand_prop           = np.array([fv(r, "prop_ms")               for r in rand_rows])
        rand_amf            = np.array([fv(r, "amf_ms")                for r in rand_rows])
        rand_smf            = np.array([fv(r, "smf_ms")                for r in rand_rows])
        rand_upf            = np.array([fv(r, "upf_ms")                for r in rand_rows])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 28))
    fig.suptitle(
        "NTN Dispatcher — Online Learning Results  (Bregman vs Random Baseline)",
        fontsize=13, fontweight="bold"
    )
    gs = fig.add_gridspec(5, 2, hspace=0.48, wspace=0.32)

    ax_prob  = fig.add_subplot(gs[0, 0])   # 1.   π_ISL / π_GND
    ax_cost  = fig.add_subplot(gs[0, 1])   # 2.   Access cost signals
    ax_reg_ll= fig.add_subplot(gs[1, 0])   # A-L. Regret log-log
    ax_reg_rt= fig.add_subplot(gs[1, 1])   # A-R. Regret / √T
    ax_del   = fig.add_subplot(gs[2, 0])   # 3.   E2E delay (rolling mean+std)
    ax_reg   = fig.add_subplot(gs[2, 1])   # 6.   Regret decomposition
    ax_task  = fig.add_subplot(gs[3, 0])   # 5.   Per-task latency bars
    ax_upf   = fig.add_subplot(gs[3, 1])   # 7.   ON-UPF instance weights
    ax_bon   = fig.add_subplot(gs[4, 0])   # B-L. ON  B_mult convergence
    ax_bgnd  = fig.add_subplot(gs[4, 1])   # B-R. GND B_mult convergence

    # ── 1. PathScheduler — π_ISL / π_GND ─────────────────────────────────────
    ax_prob.set_title("PathScheduler — Learned π_ISL / π_GND  (Level-1 Bregman)")
    ax_prob.set_xlabel("Handover index")
    ax_prob.set_ylabel("Path probability")

    ax_prob.fill_between(ho_ids, path_p_isl, alpha=0.12, color=C_ISL)
    ax_prob.fill_between(ho_ids, path_p_gnd, alpha=0.12, color=C_GND)
    ax_prob.plot(ho_ids, path_p_isl, color=C_ISL, lw=1.8, label="π_ISL (learned)")
    ax_prob.plot(ho_ids, path_p_gnd, color=C_GND, lw=1.8, label="π_GND (learned)")
    ax_prob.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5,
                    label="Uniform (random baseline)")

    add_task_shading(ax_prob, ho_ids, task_types)
    ax_prob.set_ylim(0, 1.12)
    ax_prob.legend(fontsize=8, loc="upper right")
    ax_prob.grid(True, alpha=0.3)

    # ── 2. Access cost signals ────────────────────────────────────────────────
    ax_cost.set_title("Access Cost Signals — ISL vs GND  (Level-1 Bregman input)")
    ax_cost.set_xlabel("Handover index")
    ax_cost.set_ylabel("Access cost (ms)")

    ax_cost.plot(ho_ids, acc_cost_isl, color=C_ISL, lw=0.9, alpha=0.6,
                 label="cost_ISL (ms)")
    ax_cost.plot(ho_ids, acc_cost_gnd, color=C_GND, lw=0.9, alpha=0.6,
                 label="cost_GND (ms)")

    win_c = max(1, min(15, n // 8))
    _, isl_rm = rolling_mean_valid(acc_cost_isl, 1e9, win_c)
    _, gnd_rm = rolling_mean_valid(acc_cost_gnd, 1e9, win_c)
    x_rm = np.arange(len(isl_rm))
    if len(x_rm) > 1:
        ax_cost.plot(ho_ids[x_rm], isl_rm, color=C_ISL, lw=1.8,
                     label=f"ISL avg (w={win_c})")
        ax_cost.plot(ho_ids[x_rm], gnd_rm, color=C_GND, lw=1.8,
                     label=f"GND avg (w={win_c})")

    add_task_shading(ax_cost, ho_ids, task_types)
    ax_cost.legend(fontsize=7.5, ncol=2)
    ax_cost.grid(True, alpha=0.3)

    # ── A-L. Cumulative regret — log-log ──────────────────────────────────────
    ax_reg_ll.set_xlabel("T  (handover index, log scale)")
    ax_reg_ll.set_ylabel("Cumulative global regret ms  (log scale)")

    valid_b = cum_global_reg > 0
    if valid_b.any():
        ax_reg_ll.loglog(ho_ids[valid_b], cum_global_reg[valid_b],
                         color=C_ISL, lw=2.2, label="Bregman")
    if has_rand:
        valid_r = rand_cum_reg > 0
        if valid_r.any():
            ax_reg_ll.loglog(rand_ho_ids[valid_r], rand_cum_reg[valid_r],
                             color=C_RAND, lw=1.8, ls="--", label="Random")

    # Reference growth curves anchored at first valid Bregman point
    if valid_b.sum() > 2:
        t0    = ho_ids[valid_b][0]
        r0    = cum_global_reg[valid_b][0]
        t_ref = ho_ids[valid_b]
        ax_reg_ll.loglog(t_ref, r0 * (t_ref / t0),
                         color="#E53935", ls=":", lw=1.3, label="O(T) — linear")
        ax_reg_ll.loglog(t_ref, r0 * np.sqrt(t_ref / t0),
                         color="#FB8C00", ls=":", lw=1.3, label="O(√T)")
        ax_reg_ll.loglog(t_ref, r0 * np.log(t_ref / t0 + 1),
                         color="#43A047", ls=":", lw=1.3, label="O(log T)")

        # Fit slope on second half only to avoid warmup bias
        half     = valid_b.sum() // 2
        log_t_fit = np.log(ho_ids[valid_b][half:])
        log_r_fit = np.log(cum_global_reg[valid_b][half:])
        title_str = "Cumulative Regret — Log-Log"
        if len(log_t_fit) > 5:
            slope, _ = np.polyfit(log_t_fit, log_r_fit, 1)
            verdict  = "sublinear ✓" if slope < 0.95 else "linear or super-linear ✗"
            title_str = (f"Cumulative Regret — Log-Log\n"
                         f"Fitted slope (2nd half) = {slope:.3f}  [{verdict}]")
        ax_reg_ll.set_title(title_str, fontsize=9)
    else:
        ax_reg_ll.set_title("Cumulative Regret — Log-Log", fontsize=9)

    ax_reg_ll.legend(fontsize=8)
    ax_reg_ll.grid(True, which="both", alpha=0.3)

    # ── A-R. Regret / √T ──────────────────────────────────────────────────────
    ax_reg_rt.set_title(
        "Regret / √T\n(flat or decreasing = O(√T) confirmed)",
        fontsize=9
    )
    ax_reg_rt.set_xlabel("T  (handover index)")
    ax_reg_rt.set_ylabel("Cumulative regret / √T")

    sqrt_T      = np.sqrt(ho_ids)
    norm_b      = np.where(sqrt_T > 0, cum_global_reg / sqrt_T, 0.0)
    ax_reg_rt.fill_between(ho_ids, norm_b, alpha=0.15, color=C_ISL)
    ax_reg_rt.plot(ho_ids, norm_b, color=C_ISL, lw=2.0, label="Bregman / √T")

    if has_rand:
        sqrt_Tr = np.sqrt(rand_ho_ids)
        norm_r  = np.where(sqrt_Tr > 0, rand_cum_reg / sqrt_Tr, 0.0)
        ax_reg_rt.plot(rand_ho_ids, norm_r, color=C_RAND, lw=1.8, ls="--",
                       label="Random / √T")

    # Trend annotation on second half
    if len(norm_b) > 20:
        half_n  = len(norm_b) // 2
        recent  = norm_b[half_n:]
        s_norm, _ = np.polyfit(np.arange(len(recent)), recent, 1)
        if s_norm < -0.5:
            trend = "↓ better than O(√T)"
        elif s_norm < 0.5:
            trend = "→ ≈ O(√T) ✓"
        else:
            trend = "↑ worse than O(√T) ✗"
        ax_reg_rt.annotate(
            f"Trend: {trend}",
            xy=(0.97, 0.94), xycoords="axes fraction",
            ha="right", va="top", fontsize=9, color="darkblue",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

    add_task_shading(ax_reg_rt, ho_ids, task_types)
    ax_reg_rt.set_ylim(bottom=0)
    ax_reg_rt.legend(fontsize=8)
    ax_reg_rt.grid(True, alpha=0.3)

    # ── 3. E2E delay — rolling mean ± 1σ band ────────────────────────────────
    WIN_B = max(10, n // 15)
    bx, by, bstd = rolling_nan_stats(total_ms, OUTLIER_THRESH, WIN_B)
    n_outliers   = int((total_ms >= OUTLIER_THRESH).sum())

    ax_del.set_title(
        f"End-to-End Delay — Rolling Mean ± 1σ  (w={WIN_B})"
        + (f"  [{n_outliers} outliers ≥ {OUTLIER_THRESH:.0f} ms excluded]"
           if n_outliers else ""),
        fontsize=9
    )
    ax_del.set_xlabel("Handover index")
    ax_del.set_ylabel("Total delay (ms)")

    if has_rand:
        WIN_R = max(10, nr // 15)
        rx, ry, _ = rolling_nan_stats(rand_total, OUTLIER_THRESH, WIN_R)
        ax_del.plot(rand_ho_ids[rx], ry, color=C_RAND2, lw=1.5, ls="--",
                    label=f"Random mean (w={WIN_R})")

    ax_del.fill_between(ho_ids[bx],
                        np.maximum(by - bstd, 0),
                        by + bstd,
                        alpha=0.18, color=C_ISL)
    ax_del.plot(ho_ids[bx], by, color=C_ISL, lw=2.0,
                label=f"Bregman mean ± 1σ (w={WIN_B})")

    oracle_clipped = np.minimum(oracle, CLIP_MS)
    ax_del.plot(ho_ids, oracle_clipped, color=C_BEST, lw=1.2, ls="--",
                label="Global oracle")

    add_task_shading(ax_del, ho_ids, task_types)
    ax_del.set_ylim(bottom=0, top=CLIP_MS * 1.15)
    ax_del.legend(fontsize=8)
    ax_del.grid(True, alpha=0.3)

    # ── 6. Cumulative regret decomposition ────────────────────────────────────
    ax_reg.set_title("Cumulative Regret Decomposition  (access + NF instance)")
    ax_reg.set_xlabel("Handover index")
    ax_reg.set_ylabel("Cumulative regret (ms)")

    ax_reg.fill_between(ho_ids, cum_global_reg, alpha=0.18, color="black", zorder=2)
    ax_reg.fill_between(ho_ids, cum_inst_reg,   alpha=0.30, color="#1565C0",
                        label="Bregman inst regret", zorder=3)
    ax_reg.fill_between(ho_ids, cum_access_reg, alpha=0.50, color="#FF5722",
                        label="Bregman access regret", zorder=4)
    ax_reg.plot(ho_ids, cum_global_reg, color="black", lw=1.5,
                label="Bregman global regret", zorder=5)

    if has_rand:
        ax_reg.plot(rand_ho_ids, rand_cum_reg, color=C_RAND2, lw=1.5,
                    ls="--", label="Random global regret", zorder=5)
        ax_reg.plot(rand_ho_ids, rand_cum_access_reg, color=C_GND, lw=0.9,
                    ls=":", alpha=0.6, label="Random access regret", zorder=4)
        ax_reg.plot(rand_ho_ids, rand_cum_inst_reg, color=C_ISL, lw=0.9,
                    ls=":", alpha=0.6, label="Random inst regret", zorder=4)

    add_task_shading(ax_reg, ho_ids, task_types)
    ax_reg.legend(fontsize=8, ncol=2)
    ax_reg.grid(True, alpha=0.3)

    # ── 5. Per-task mean latency — Bregman vs Random ──────────────────────────
    ax_task.set_title(
        f"Mean Total Latency by Task Type  (non-outlier HOs < {OUTLIER_THRESH:.0f} ms)"
    )
    ax_task.set_xlabel("Task type")
    ax_task.set_ylabel("Mean total delay (ms)")

    COMP_SPECS = [
        ("prop_ms", "Propagation", "#90CAF9", 0.95),
        ("amf_ms",  "AMF",         "#1565C0", 0.90),
        ("smf_ms",  "SMF",         "#FF8A65", 0.90),
        ("upf_ms",  "UPF",         "#BF360C", 0.90),
    ]
    comp_keys = [c[0] for c in COMP_SPECS]

    task_bregman: dict[str, dict[str, list[float]]] = {
        tt: {k: [] for k in comp_keys} for tt in TASK_ORDER
    }
    for i, tt in enumerate(task_types):
        if total_ms[i] < OUTLIER_THRESH:
            for key, arr in [("prop_ms", prop_ms), ("amf_ms", amf_ms),
                              ("smf_ms", smf_ms), ("upf_ms", upf_ms)]:
                task_bregman[tt][key].append(float(arr[i]))

    task_rand: dict[str, dict[str, list[float]]] = {
        tt: {k: [] for k in comp_keys} for tt in TASK_ORDER
    }
    if has_rand:
        for i, tt in enumerate(rand_task_types):
            if rand_total[i] < OUTLIER_THRESH:
                for key, arr in [("prop_ms", rand_prop), ("amf_ms", rand_amf),
                                  ("smf_ms", rand_smf), ("upf_ms", rand_upf)]:
                    task_rand[tt][key].append(float(arr[i]))

    x_task = np.arange(len(TASK_ORDER))
    bar_w  = 0.35

    for gi, tt in enumerate(TASK_ORDER):
        b_n = len(task_bregman[tt]["prop_ms"])
        r_n = len(task_rand[tt]["prop_ms"]) if has_rand else 0
        b_total = sum(
            np.mean(task_bregman[tt][k]) if task_bregman[tt][k] else 0.0
            for k in comp_keys
        )
        r_total = sum(
            np.mean(task_rand[tt][k]) if task_rand[tt][k] else 0.0
            for k in comp_keys
        ) if has_rand else 0.0

        b_bottom = r_bottom = 0.0
        for key, lbl, clr, al in COMP_SPECS:
            b_mean = float(np.mean(task_bregman[tt][key])) if task_bregman[tt][key] else 0.0
            ax_task.bar(x_task[gi] - bar_w / 2, b_mean, bar_w,
                        bottom=b_bottom, color=clr, alpha=al,
                        label=lbl if gi == 0 else "_nolegend_",
                        edgecolor="white", lw=0.4)
            if b_mean > 0.5:
                ax_task.text(x_task[gi] - bar_w / 2, b_bottom + b_mean / 2,
                             f"{b_mean:.1f}", ha="center", va="center",
                             fontsize=5.5, color="white", fontweight="bold")
            b_bottom += b_mean

            if has_rand:
                r_mean = float(np.mean(task_rand[tt][key])) if task_rand[tt][key] else 0.0
                ax_task.bar(x_task[gi] + bar_w / 2, r_mean, bar_w,
                            bottom=r_bottom, color=clr, alpha=al * 0.45,
                            hatch="///", label="_nolegend_",
                            edgecolor="white", lw=0.4)
                if r_mean > 0.5:
                    ax_task.text(x_task[gi] + bar_w / 2, r_bottom + r_mean / 2,
                                 f"{r_mean:.1f}", ha="center", va="center",
                                 fontsize=5.5, color="#222222", fontweight="bold")
                r_bottom += r_mean

        if b_total > 0.2:
            ax_task.text(x_task[gi] - bar_w / 2, b_bottom + 0.2,
                         f"{b_total:.1f}\nn={b_n}",
                         ha="center", va="bottom", fontsize=6.5, color="#333333")
        if has_rand and r_total > 0.2:
            ax_task.text(x_task[gi] + bar_w / 2, r_bottom + 0.2,
                         f"{r_total:.1f}\nn={r_n}",
                         ha="center", va="bottom", fontsize=6.5, color="#555555")

    ax_task.set_xticks(x_task)
    ax_task.set_xticklabels([t.capitalize() for t in TASK_ORDER], fontsize=9)
    comp_handles = [mpatches.Patch(color=clr, alpha=al, label=lbl)
                    for _, lbl, clr, al in COMP_SPECS]
    bregman_h = mpatches.Patch(color="#888888", alpha=0.88, label="Bregman (solid)")
    handles = comp_handles + [bregman_h]
    if has_rand:
        rand_h = mpatches.Patch(color="#888888", alpha=0.38, hatch="///",
                                label="Random (hatched)")
        handles.append(rand_h)
    ax_task.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")
    ax_task.grid(axis="y", alpha=0.3)
    ax_task.annotate(f"Outliers >{OUTLIER_THRESH:.0f} ms excluded",
                     xy=(0.01, 0.98), xycoords="axes fraction",
                     va="top", fontsize=6.5, color="gray")

    # ── 7. ON-UPF instance weights ────────────────────────────────────────────
    ax_upf.set_title(
        "Onboard UPF Instance Selection Weights\n"
        "(ON-UPF-2 should rise at instagram — only stable NTN UPF under high load)",
        fontsize=9
    )
    ax_upf.set_xlabel("Handover index")
    ax_upf.set_ylabel("Selection probability")

    colors_upf = ["#E53935", "#FFA726", "#43A047"]
    labels_upf = [
        "ON-UPF-0  (unstable @ instagram)",
        "ON-UPF-1  (unstable @ instagram)",
        "ON-UPF-2  (only stable NTN UPF @ instagram)",
    ]
    for i, (probs, color, label) in enumerate(zip(on_upf_p, colors_upf, labels_upf)):
        lw = 2.0 if i == 2 else 1.4
        ax_upf.plot(ho_ids, probs, color=color, lw=lw, label=label)

    ax_upf.axhline(1 / 3, color="#888888", ls=":", lw=0.9, label="Uniform (1/3)")

    add_task_shading(ax_upf, ho_ids, task_types)
    ax_upf.set_ylim(-0.02, 1.12)
    ax_upf.legend(fontsize=7.5, ncol=1)
    ax_upf.grid(True, alpha=0.3)
    ax_upf.annotate(
        "GND-UPF always stable — see B_mult panels below",
        xy=(0.01, 0.02), xycoords="axes fraction",
        va="bottom", fontsize=6, color="gray"
    )

    # ── B-L/B-R. B_mult convergence — exploitation arm ────────────────────────
    if has_bmult:
        BMULT_LAYERS_ON  = [
            ("on_amf",  "AMF",  on_amf_B,  "-",  "#90CAF9", 0.35),
            ("on_smf",  "SMF",  on_smf_B,  "--", "#CE93D8", 0.35),
            ("on_upf",  "UPF",  on_upf_B,  ":",  None,      1.0 ),
        ]
        BMULT_LAYERS_GND = [
            ("gnd_amf", "AMF",  gnd_amf_B, "-",  "#90CAF9", 0.35),
            ("gnd_smf", "SMF",  gnd_smf_B, "--", "#CE93D8", 0.35),
            ("gnd_upf", "UPF",  gnd_upf_B, ":",  None,      1.0 ),
        ]
        INST_COLORS_ON  = ["#E53935", "#FFA726", "#43A047"]
        INST_COLORS_GND = ["#1565C0", "#1976D2", "#42A5F5"]

        for ax_bm, layers, inst_colors in [
            (ax_bon,  BMULT_LAYERS_ON,  INST_COLORS_ON),
            (ax_bgnd, BMULT_LAYERS_GND, INST_COLORS_GND),
        ]:
            for _prefix, nf_name, bmult_data, ls, flat_color, alpha in layers:
                is_upf = (nf_name == "UPF")
                for k, (vals, inst_color) in enumerate(zip(bmult_data, inst_colors)):
                    color = inst_color if is_upf else flat_color
                    lw    = 1.8 if is_upf else 0.7
                    label = (f"UPF inst {k}" if is_upf else "_nolegend_")
                    ax_bm.plot(ho_ids, vals, color=color, ls=ls, lw=lw,
                               alpha=alpha, label=label)

            # Reference lines
            ax_bm.axhline(1.0,        color="#888888", ls="--", lw=0.9, alpha=0.6,
                          label="B_mult = 1.0  (hardware baseline)")
            ax_bm.axhline(B_MULT_MIN, color="#E53935", ls=":", lw=0.7, alpha=0.45,
                          label=f"B_MULT_MIN ({B_MULT_MIN})")
            ax_bm.axhline(B_MULT_MAX, color="#43A047", ls=":", lw=0.7, alpha=0.45,
                          label=f"B_MULT_MAX ({B_MULT_MAX})")

            add_task_shading(ax_bm, ho_ids, task_types)
            ax_bm.set_ylim(0.3, 2.3)
            ax_bm.set_xlabel("Handover index")
            ax_bm.set_ylabel("B_mult  (exploitation multiplier)")
            ax_bm.legend(fontsize=7.5, ncol=2)
            ax_bm.grid(True, alpha=0.3)
            ax_bm.annotate(
                "Bold coloured = UPF (task-sensitive);  faint = AMF/SMF (always stable)",
                xy=(0.01, 0.02), xycoords="axes fraction",
                va="bottom", fontsize=6, color="gray"
            )

        ax_bon.set_title(
            "ON B_mult — Exploitation Arm\n"
            "(UPF-ON-2 should rise; UPF-ON-0/1 should fall at instagram blocks)",
            fontsize=9
        )
        ax_bgnd.set_title(
            "GND B_mult — Exploitation Arm\n"
            "(GND UPF always stable → smooth convergence expected)",
            fontsize=9
        )
    else:
        for ax_bm, label in [(ax_bon, "ON B_mult"), (ax_bgnd, "GND B_mult")]:
            ax_bm.text(0.5, 0.5, f"{label}\n(B_mult columns not found in CSV)",
                       ha="center", va="center", transform=ax_bm.transAxes,
                       fontsize=10, color="gray")

    # ── Summary ───────────────────────────────────────────────────────────────
    isl_count = int(isl_mask.sum())
    gnd_count = int(gnd_mask.sum())
    print(f"\n{'═'*62}")
    print(f"  Dispatcher results summary  ({n} HOs)")
    print(f"{'═'*62}")
    print(f"  Bregman ISL selected : {isl_count:>4}  ({100*isl_count/n:.1f}%)")
    print(f"  Bregman GND selected : {gnd_count:>4}  ({100*gnd_count/n:.1f}%)")

    tc = Counter(task_types)
    print(f"{'─'*62}")
    print(f"  Task type distribution:")
    for tt, cnt in sorted(tc.items()):
        print(f"    {tt:<12}: {cnt:>4}  ({100*cnt/n:.1f}%)")

    print(f"{'─'*62}")
    print(f"  Per-task mean latency breakdown (non-outlier < {OUTLIER_THRESH:.0f} ms):")
    for tt in TASK_ORDER:
        b_comps = task_bregman.get(tt, {})
        b_n     = len(b_comps.get("prop_ms", []))
        b_tot   = sum(np.mean(b_comps[k]) if b_comps.get(k) else 0.0 for k in comp_keys)
        b_str   = f"{b_tot:.2f} ms (n={b_n})" if b_n > 0 else "—"
        r_comps = task_rand.get(tt, {}) if has_rand else {}
        r_n     = len(r_comps.get("prop_ms", []))
        r_tot   = sum(np.mean(r_comps[k]) if r_comps.get(k) else 0.0 for k in comp_keys)
        r_str   = f"{r_tot:.2f} ms (n={r_n})" if r_n > 0 else "—"
        print(f"    {tt:<12}: Bregman={b_str}  Random={r_str}")

    print(f"{'─'*62}")
    for label, mask in [("ISL (Bregman)", isl_mask), ("GND (Bregman)", gnd_mask)]:
        if not mask.any():
            continue
        t = total_ms[mask]
        print(f"  {label}: avg={t.mean():.2f} ms  min={t.min():.2f}  max={t.max():.2f}")

    if has_rand:
        print(f"{'─'*62}")
        ri = int(rand_isl_mask.sum())
        rg = int(rand_gnd_mask.sum())
        print(f"  Random  ISL selected : {ri:>4}  ({100*ri/nr:.1f}%)")
        print(f"  Random  GND selected : {rg:>4}  ({100*rg/nr:.1f}%)")

    print(f"{'─'*62}")
    oracle_arr = np.array([fv(r, "global_oracle_ms") for r in rows])
    print(f"  Global oracle avg      : {oracle_arr.mean():.2f} ms")
    print(f"  Bregman access regret  : avg={cum_access_reg[-1]/n:.2f} ms  "
          f"(cum={cum_access_reg[-1]:.1f} ms)")
    print(f"  Bregman inst   regret  : avg={cum_inst_reg[-1]/n:.2f} ms  "
          f"(cum={cum_inst_reg[-1]:.1f} ms)")
    print(f"  Bregman global regret  : avg={cum_global_reg[-1]/n:.2f} ms  "
          f"(cum={cum_global_reg[-1]:.1f} ms)")
    if has_rand and nr > 0:
        print(f"  Random  global regret  : avg={rand_cum_reg[-1]/nr:.2f} ms  "
              f"(cum={rand_cum_reg[-1]:.1f} ms)")
    print(f"{'═'*62}\n")

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {OUT_PNG}")

    # ── Debug figure (saved separately, always) ───────────────────────────────
    save_debug_instagram(rows, ho_ids, task_types)
    save_level1_dynamics(rows, ho_ids, task_types)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
