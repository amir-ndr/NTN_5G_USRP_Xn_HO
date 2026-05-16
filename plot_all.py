#!/usr/bin/env python3
"""
plot_all.py — Generate every experimental plot for the paper.

Reads:
  results/<tau>_ls<scale>/dispatch_log_<tag>.csv     (Bregman dispatcher)
  results/<tau>_ls<scale>/random_log_<tag>.csv       (Random baseline)
  results/<tau>_ls<scale>/greedy_log_<tag>.csv       (Greedy full-info oracle, optional)

Outputs:
  results_all/exp1/  Regret convergence
  results_all/exp2/  Instance probability trajectories
  results_all/exp3/  Task-type sensitivity
  results_all/exp4/  Load sweep (requires runs at ls=0.5,1.0,1.5,2.0)
  results_all/exp5/  τ_max sensitivity (requires strict/default/relaxed)
  results_all/exp6/  Path selection vs orbital geometry
  results_all/exp7/  Baseline comparisons (Bregman vs Random vs in-data oracle)

Required runs (controller.py invocations) for full coverage:
  python3 controller.py --tau-profile default --load-scale 0.5 --ho-every 5
  python3 controller.py --tau-profile default --load-scale 1.0 --ho-every 5
  python3 controller.py --tau-profile default --load-scale 1.5 --ho-every 5
  python3 controller.py --tau-profile default --load-scale 2.0 --ho-every 5
  python3 controller.py --tau-profile strict  --load-scale 1.0 --ho-every 5
  python3 controller.py --tau-profile relaxed --load-scale 1.0 --ho-every 5

Usage:
  python3 plot_all.py                 # plot every available experiment
  python3 plot_all.py --exp 1 2 6     # only specified experiments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUT_DIR     = HERE / "results_all"

# ── style ──────────────────────────────────────────────────────────────────────
C_BREG   = "#2196F3"   # Bregman dispatcher
C_RAND   = "#9E9E9E"   # Random baseline
C_GREEDY = "#7B1FA2"   # Greedy full-info oracle (single-instance upper bound)
C_ORACLE = "#4CAF50"   # Optimal-split oracle (lower bound)

PROFILE_COLOR = {"default": "#2196F3", "strict": "#F44336", "relaxed": "#4CAF50"}
PROFILE_LABEL = {"default": "Default τ", "strict": "Strict τ", "relaxed": "Relaxed τ"}

INST_COLORS = ["#1565C0", "#EF6C00", "#2E7D32"]

TASK_COLORS = {"gaming": "#E91E63", "youtube": "#FF5722", "browsing": "#4CAF50",
               "instagram": "#9C27B0", "mixed": "#FFC107"}
TASK_ORDER = ["gaming", "youtube", "browsing", "instagram", "mixed"]

LOAD_COLORS = {0.5: "#90CAF9", 1.0: "#1976D2", 1.5: "#F57C00", 2.0: "#C62828"}

plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "lines.linewidth": 1.6,
    "savefig.bbox":    "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ls_tag(ls: float) -> str:
    return f"{ls:g}".replace(".", "p")


def _run_tag(tau: str, ls: float) -> str:
    return f"{tau}_ls{_ls_tag(ls)}"


def _csv(tau: str, ls: float, kind: str) -> Path:
    tag = _run_tag(tau, ls)
    return RESULTS_DIR / tag / f"{kind}_{tag}.csv"


def _load(tau: str, ls: float, kind: str = "dispatch_log") -> pd.DataFrame | None:
    p = _csv(tau, ls, kind)
    if not p.exists():
        print(f"  [skip] missing: {p.relative_to(HERE)}")
        return None
    return pd.read_csv(p)


def _layer_entropy(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Shannon entropy of an instance-probability triple {prefix}_p0/p1/p2 per HO."""
    cols = [f"{prefix}_p{k}" for k in range(3)]
    if not all(c in df.columns for c in cols):
        return np.array([])
    p = df[cols].to_numpy()
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _task_switch_idx(df: pd.DataFrame) -> list[int]:
    if "task_type" not in df.columns:
        return []
    tt = df["task_type"].to_numpy()
    return [i for i in range(1, len(tt)) if tt[i] != tt[i - 1]]


def _annotate_task_switches(ax, df: pd.DataFrame) -> None:
    for i in _task_switch_idx(df):
        ax.axvline(df["ho_id"].iloc[i], color="black", lw=0.4, alpha=0.25)


def _add_task_background(ax, df: pd.DataFrame) -> None:
    """Add task-type background shading with labels on top."""
    if "task_type" not in df.columns:
        return

    task_types = df["task_type"].tolist()
    ho_ids = df["ho_id"].to_numpy()

    if not task_types:
        return

    # Build task blocks
    blocks = []
    prev_tt = task_types[0]
    start_x = ho_ids[0]

    for i in range(1, len(ho_ids)):
        if task_types[i] != prev_tt:
            blocks.append((start_x, ho_ids[i - 1], prev_tt))
            start_x = ho_ids[i]
            prev_tt = task_types[i]
    blocks.append((start_x, ho_ids[-1], prev_tt))

    # Draw blocks
    task_short = {"gaming": "GAM", "youtube": "YT", "browsing": "BR",
                  "instagram": "IG", "mixed": "MX"}
    task_colors = {"gaming": "#BBDEFB", "youtube": "#FFCDD2", "browsing": "#C8E6C9",
                   "instagram": "#FFE0B2", "mixed": "#E1BEE7"}

    ymin, ymax = ax.get_ylim()
    for x0, x1, tt in blocks:
        ax.axvspan(float(x0) - 0.5, float(x1) + 0.5,
                   alpha=0.08, color=task_colors.get(tt, "#F5F5F5"), zorder=0)

    # Add labels on top
    for x0, x1, tt in blocks:
        mid = (float(x0) + float(x1)) / 2
        ax.text(mid, 0.98, task_short.get(tt, tt[:3].upper()),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="#333333", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.7, edgecolor="none"))


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Regret Convergence
# ══════════════════════════════════════════════════════════════════════════════

def exp1_regret(out: Path) -> None:
    breg = _load("default", 1.0, "dispatch_log")
    rand = _load("default", 1.0, "random_log")
    grdy = _load("default", 1.0, "greedy_log")
    if breg is None or rand is None:
        return

    def _cum_global(df: pd.DataFrame) -> np.ndarray:
        if "cum_global_regret_ms" in df.columns:
            return df["cum_global_regret_ms"].to_numpy()
        return df["global_regret_ms"].cumsum().to_numpy()

    rand_cum_global = _cum_global(rand)

    # ── 1a: cumulative global regret ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(breg["ho_id"], breg["cum_global_regret_ms"],
            color=C_BREG, label="Bregman (Two-Level)")
    ax.plot(rand["ho_id"], rand_cum_global,
            color=C_RAND, linestyle="--", label="Random baseline")
    if grdy is not None:
        ax.plot(grdy["ho_id"], _cum_global(grdy),
                color=C_GREEDY, linestyle="-.", label="Greedy full-info oracle")
    ax.set_xlabel("Handover index")
    ax.set_ylabel("Cumulative global regret (ms)")
    ax.set_title("Cumulative Global Regret vs Handover Index")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "1a_cumulative_regret.png"); plt.close(fig)

    # ── 1b: time-averaged regret on log–log axes ─────────────────────────────
    ho_b = breg["ho_id"].to_numpy()
    ho_r = rand["ho_id"].to_numpy()
    avg_b = breg["cum_global_regret_ms"].to_numpy() / np.maximum(ho_b, 1)
    avg_r = rand_cum_global                          / np.maximum(ho_r, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(ho_b, avg_b, color=C_BREG, label="Bregman")
    ax.loglog(ho_r, avg_r, color=C_RAND, linestyle="--", label="Random")
    if grdy is not None:
        ho_g  = grdy["ho_id"].to_numpy()
        avg_g = _cum_global(grdy) / np.maximum(ho_g, 1)
        ax.loglog(ho_g, avg_g, color=C_GREEDY, linestyle="-.", label="Greedy")
    # T^{-1/2} reference line anchored at first Bregman point
    ref_y0 = avg_b[max(1, len(avg_b)//10)]
    ref_x0 = ho_b[max(1, len(avg_b)//10)]
    ref = ref_y0 * np.sqrt(ref_x0 / np.maximum(ho_b, 1))
    ax.loglog(ho_b, ref, color="black", linestyle=":", lw=1.0,
              label=r"$T^{-1/2}$ reference")
    ax.set_xlabel("Handover index  T  (log)")
    ax.set_ylabel(r"Time-averaged regret  $R(T)/T$  (ms, log)")
    ax.set_title("Time-Averaged Regret (log–log)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig(out / "1b_time_averaged_regret_loglog.png"); plt.close(fig)

    # ── 1c: decomposed regret ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, color, label in [
        ("cum_access_regret_ms", "#1976D2", "Access regret (Level-1)"),
        ("cum_inst_regret_ms",   "#F57C00", "Instance regret (Level-2)"),
        ("cum_global_regret_ms", "#2E7D32", "Global regret"),
    ]:
        if col in breg.columns:
            ax.plot(breg["ho_id"], breg[col], color=color, label=label)
    ax.set_xlabel("Handover index")
    ax.set_ylabel("Cumulative regret (ms)")
    ax.set_title("Regret Decomposition (Bregman)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "1c_regret_decomposition.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Instance Probability Trajectories
# ══════════════════════════════════════════════════════════════════════════════

def exp2_probabilities(out: Path) -> None:
    df = _load("default", 1.0, "dispatch_log")
    if df is None:
        return

    # 2a: 2×3 panel (rows AMF/SMF/UPF, cols ON/GND) ── wait, the user spec is
    # 2 rows (AMF, SMF, UPF stacked) × 2 cols (ON, GND). We do AMF/SMF/UPF rows ×
    # NTN/TN cols → 3×2 = 6 panels.
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    fig.suptitle("Instance Selection Probability — Default τ, load = 1.0",
                 fontsize=13, y=1.0)
    layers = [("amf", "AMF"), ("smf", "SMF"), ("upf", "UPF")]
    sides  = [("on", "NTN (onboard)"), ("gnd", "TN (ground)")]
    for row, (lay, lay_label) in enumerate(layers):
        for col, (side, side_label) in enumerate(sides):
            ax = axes[row, col]
            cols = [f"{side}_{lay}_p{k}" for k in range(3)]
            if not all(c in df.columns for c in cols):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                for k in range(3):
                    ax.plot(df["ho_id"], df[cols[k]], color=INST_COLORS[k],
                            label=f"Instance {k}", alpha=0.9)
                _annotate_task_switches(ax, df)
            if row == 0: ax.set_title(side_label)
            if col == 0: ax.set_ylabel(f"{lay_label}\nselection prob.")
            if row == 2: ax.set_xlabel("Handover index")
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "2a_instance_probability_grid.png"); plt.close(fig)

    # 2b: UPF probabilities broken out by task (ISL/ON path) ─────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    fig.suptitle("UPF (NTN) Selection Probability per Task Type",
                 fontsize=13, y=1.02)
    for ax, task in zip(axes, TASK_ORDER):
        sub = df[df["task_type"] == task]
        if sub.empty:
            ax.text(0.5, 0.5, f"no {task}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(task); continue
        for k in range(3):
            col = f"on_upf_p{k}"
            if col in sub.columns:
                ax.plot(sub["ho_id"], sub[col], color=INST_COLORS[k],
                        label=f"UPF-ON-{k}", alpha=0.9)
        ax.set_title(task)
        ax.set_xlabel("Handover index")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Selection probability")
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "2b_upf_ntn_probs_per_task.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Task-Type Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def exp3_task_sensitivity(out: Path) -> None:
    breg = _load("default", 1.0, "dispatch_log")
    rand = _load("default", 1.0, "random_log")
    if breg is None or rand is None:
        return

    # 3a: latency box plots per task, Bregman vs Random ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    positions, labels, data, colors = [], [], [], []
    width = 0.4
    for i, task in enumerate(TASK_ORDER):
        b = breg.loc[breg["task_type"] == task, "total_ms"].dropna()
        r = rand.loc[rand["task_type"] == task, "total_ms"].dropna()
        positions.extend([i - width/2, i + width/2])
        data.extend([b.values, r.values])
        colors.extend([C_BREG, C_RAND])
        labels.append(task)

    bp = ax.boxplot(data, positions=positions, widths=width*0.95,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black"); median.set_linewidth(1.5)

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("End-to-end latency total_ms (ms)")
    ax.set_title("Latency Distribution per Task — Bregman vs Random")
    ax.grid(True, axis="y", alpha=0.3)
    # synthetic legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=C_BREG, alpha=0.7),
               Patch(facecolor=C_RAND, alpha=0.7)],
              ["Bregman", "Random"], loc="upper left")
    fig.savefig(out / "3a_latency_box_per_task.png"); plt.close(fig)

    # 3b: UPF instance choice histogram per task (ISL/ON path only) ──────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharey="row")
    fig.suptitle("UPF Instance Selection Frequency per Task (chosen path only)",
                 fontsize=13, y=1.0)
    for col, task in enumerate(TASK_ORDER):
        for row, (side, side_label) in enumerate([("ISL", "ISL (NTN UPF)"),
                                                  ("GND", "GND (TN UPF)")]):
            ax = axes[row, col]
            sub = breg[(breg["task_type"] == task) & (breg["path"] == side)]
            counts = sub["upf_inst"].value_counts().reindex([0, 1, 2], fill_value=0)
            bars = ax.bar(["Inst 0", "Inst 1", "Inst 2"], counts.values,
                          color=INST_COLORS, edgecolor="white")
            for b, c in zip(bars, counts.values):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                        str(int(c)), ha="center", va="bottom", fontsize=9)
            if row == 0: ax.set_title(task)
            if col == 0: ax.set_ylabel(f"{side_label}\ncount")
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "3b_upf_choice_histogram_per_task.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 4 — Load Sweep
# ══════════════════════════════════════════════════════════════════════════════

def exp4_load_sweep(out: Path) -> None:
    scales = [0.5, 1.0, 1.5, 2.0]
    data: dict[float, dict[str, pd.DataFrame]] = {}
    for ls in scales:
        breg = _load("default", ls, "dispatch_log")
        rand = _load("default", ls, "random_log")
        if breg is not None and rand is not None:
            data[ls] = {"breg": breg, "rand": rand}
    if not data:
        return

    # 4a: mean total_ms / core_ms vs load, post-convergence (last 100 HOs) ────
    POST = 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, ylabel in [(axes[0], "total_ms", "Mean end-to-end latency (ms)"),
                               (axes[1], "core_ms",  "Mean core compute latency (ms)")]:
        xs = sorted(data.keys())
        breg_mean = [data[ls]["breg"][metric].iloc[-POST:].mean() for ls in xs]
        breg_std  = [data[ls]["breg"][metric].iloc[-POST:].std()  for ls in xs]
        rand_mean = [data[ls]["rand"][metric].iloc[-POST:].mean() for ls in xs]
        rand_std  = [data[ls]["rand"][metric].iloc[-POST:].std()  for ls in xs]
        ax.errorbar(xs, breg_mean, yerr=breg_std, color=C_BREG, marker="o",
                    capsize=3, label="Bregman")
        ax.errorbar(xs, rand_mean, yerr=rand_std, color=C_RAND, marker="s",
                    linestyle="--", capsize=3, label="Random")
        ax.set_xlabel("Load scale (--load-scale)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_title("Latency vs Load")
    axes[1].set_title("Core Compute vs Load")
    fig.tight_layout()
    fig.savefig(out / "4a_latency_vs_load.png"); plt.close(fig)

    # 4b: ISL fraction stacked bar per load scale ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = sorted(data.keys())
    isl_frac = []
    for ls in xs:
        df = data[ls]["breg"]
        isl_frac.append((df["path"] == "ISL").mean())
    isl_frac = np.array(isl_frac)
    gnd_frac = 1 - isl_frac
    x = np.arange(len(xs))
    ax.bar(x, isl_frac, color=C_BREG, label="ISL")
    ax.bar(x, gnd_frac, bottom=isl_frac, color="#FF8A65", label="GND")
    for i, (li, gi) in enumerate(zip(isl_frac, gnd_frac)):
        ax.text(i, li/2, f"{li*100:.0f}%", ha="center", va="center",
                color="white", fontweight="bold")
        if gi > 0.05:
            ax.text(i, li + gi/2, f"{gi*100:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(ls) for ls in xs])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Load scale"); ax.set_ylabel("Path selection fraction")
    ax.set_title("Path Selection vs Load (Bregman)")
    ax.legend(loc="lower right")
    fig.savefig(out / "4b_path_fraction_vs_load.png"); plt.close(fig)

    # 4c: unstable-instance heatmap per load — ON UPF only (most variable) ────
    fig, axes = plt.subplots(1, len(xs), figsize=(4*len(xs), 4),
                             sharey=True, squeeze=False)
    fig.suptitle("Fraction of HOs with chosen UPF instance ρ ≥ 0.95  "
                 "(higher = more unstable selections)", fontsize=12, y=1.03)
    for ax, ls in zip(axes[0], xs):
        df = data[ls]["breg"]
        mat = np.zeros((len(TASK_ORDER), 3))
        for r, task in enumerate(TASK_ORDER):
            sub = df[(df["task_type"] == task) & (df["path"] == "ISL")]
            if sub.empty:
                continue
            for k in range(3):
                chosen_k = sub[sub["upf_inst"] == k]
                if not chosen_k.empty:
                    mat[r, k] = (chosen_k["upf_rho"] >= 0.95).mean()
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["UPF-0", "UPF-1", "UPF-2"])
        ax.set_yticks(range(len(TASK_ORDER))); ax.set_yticklabels(TASK_ORDER)
        ax.set_title(f"load = {ls}")
        for r in range(len(TASK_ORDER)):
            for c in range(3):
                ax.text(c, r, f"{mat[r, c]*100:.0f}%", ha="center", va="center",
                        color="white" if mat[r, c] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=axes[0, -1], shrink=0.85, label="unstable fraction")
    fig.tight_layout()
    fig.savefig(out / "4c_unstable_upf_heatmap.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 5 — τ_max Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def exp5_tau_sensitivity(out: Path) -> None:
    profiles = ["strict", "default", "relaxed"]
    data = {p: _load(p, 1.0, "dispatch_log") for p in profiles}
    data = {p: df for p, df in data.items() if df is not None}
    if not data:
        return

    # 5a: cumulative regret ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, df in data.items():
        ax.plot(df["ho_id"], df["cum_global_regret_ms"],
                color=PROFILE_COLOR[p], label=PROFILE_LABEL[p])
    ax.set_xlabel("Handover index"); ax.set_ylabel("Cumulative global regret (ms)")
    ax.set_title("Regret vs τ_max Profile")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "5a_regret_vs_tau_profile.png"); plt.close(fig)

    # 5b: average entropy across 6 NF layers ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layer_prefixes = ["on_amf", "on_smf", "on_upf",
                      "gnd_amf", "gnd_smf", "gnd_upf"]
    for p, df in data.items():
        ents = []
        for pre in layer_prefixes:
            e = _layer_entropy(df, pre)
            if e.size > 0:
                ents.append(e)
        if not ents:
            continue
        mean_ent = np.mean(np.stack(ents, axis=0), axis=0)
        ax.plot(df["ho_id"], mean_ent, color=PROFILE_COLOR[p],
                label=PROFILE_LABEL[p])
    ax.axhline(np.log(3), color="black", linestyle=":", lw=1,
               label="Uniform (log 3)")
    ax.set_xlabel("Handover index"); ax.set_ylabel("Mean entropy across NF layers (nats)")
    ax.set_title("Selection Decisiveness over Time  (lower = more decisive)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "5b_entropy_vs_tau_profile.png"); plt.close(fig)

    # 5c: final probability bar chart for UPF (ISL/ON side), per profile ────
    fig, axes = plt.subplots(1, len(data), figsize=(4*len(data), 4),
                             sharey=True, squeeze=False)
    fig.suptitle("Final UPF-ON Selection Distribution (mean over last 50 HOs)",
                 fontsize=12, y=1.03)
    for ax, (p, df) in zip(axes[0], data.items()):
        last = df.iloc[-50:]
        bars = [last[f"on_upf_p{k}"].mean() for k in range(3)
                if f"on_upf_p{k}" in df.columns]
        if not bars:
            continue
        ax.bar(["UPF-0", "UPF-1", "UPF-2"], bars,
               color=INST_COLORS, edgecolor="white")
        for i, v in enumerate(bars):
            ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_title(PROFILE_LABEL[p])
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Selection probability")
    fig.tight_layout()
    fig.savefig(out / "5c_final_upf_distribution.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 6 — Orbital Geometry vs Path Selection
# ══════════════════════════════════════════════════════════════════════════════

def exp6_geometry(out: Path) -> None:
    df = _load("default", 1.0, "dispatch_log")
    if df is None:
        return

    # 6a: stacked 3-panel (delays, π_ISL, chosen path scatter) ───────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    ax0, ax1, ax2 = axes

    ax0.plot(df["ho_id"], df["isl_ms"], color=C_BREG, label="ISL prop delay")
    ax0.plot(df["ho_id"], df["gnd_ms"], color="#FF8A65", label="GND prop delay")
    ax0.set_ylabel("Propagation delay (ms)")
    ax0.set_title("Orbital Geometry vs Level-1 Path Selection")
    ax0.grid(True, alpha=0.3); ax0.legend(loc="upper right")

    ax1.plot(df["ho_id"], df["path_p_isl"], color=C_BREG, label=r"$\pi_{ISL}$")
    ax1.axhline(0.5, color="black", linestyle=":", lw=1)
    ax1.set_ylabel(r"Algorithm $\pi_{ISL}$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.grid(True, alpha=0.3)

    chose_isl = df["path"] == "ISL"
    ax2.scatter(df.loc[chose_isl, "ho_id"], [1]*int(chose_isl.sum()),
                color=C_BREG, marker="|", s=40, label="ISL chosen")
    ax2.scatter(df.loc[~chose_isl, "ho_id"], [0]*int((~chose_isl).sum()),
                color="#FF8A65", marker="|", s=40, label="GND chosen")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["GND", "ISL"])
    ax2.set_xlabel("Handover index")
    ax2.set_ylim(-0.3, 1.3)
    ax2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "6a_geometry_vs_path_selection.png"); plt.close(fig)

    # 6b: gap vs π_ISL scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    gap = df["isl_ms"] - df["gnd_ms"]
    sc = ax.scatter(gap, df["path_p_isl"], c=df["ho_id"],
                    cmap="viridis", s=18, alpha=0.85)
    ax.axvline(0, color="black", linestyle=":", lw=1)
    ax.axhline(0.5, color="black", linestyle=":", lw=1)
    ax.set_xlabel(r"$d_{ISL} - d_{GND}$  (ms) — negative ⇒ ISL shorter")
    ax.set_ylabel(r"$\pi_{ISL}$")
    ax.set_title(r"Path Probability vs Propagation Delay Gap")
    cbar = fig.colorbar(sc, ax=ax, label="HO index")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "6b_path_prob_vs_delay_gap.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 7 — Baseline Comparisons (uses logged global_oracle_ms)
# ══════════════════════════════════════════════════════════════════════════════

def exp7_baselines(out: Path) -> None:
    breg = _load("default", 1.0, "dispatch_log")
    rand = _load("default", 1.0, "random_log")
    grdy = _load("default", 1.0, "greedy_log")
    if breg is None or rand is None:
        return

    # 7a: cumulative latency vs HO — Bregman, Random, Greedy, optimal-split oracle ─
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(breg["ho_id"], breg["total_ms"].cumsum(),
            color=C_BREG, label="Bregman cumulative total_ms")
    ax.plot(rand["ho_id"], rand["total_ms"].cumsum(),
            color=C_RAND, linestyle="--", label="Random cumulative total_ms")
    if grdy is not None:
        ax.plot(grdy["ho_id"], grdy["total_ms"].cumsum(),
                color=C_GREEDY, linestyle="-.",
                label="Greedy full-info oracle cumulative")
    if "global_oracle_ms" in breg.columns:
        ax.plot(breg["ho_id"], breg["global_oracle_ms"].cumsum(),
                color=C_ORACLE, linestyle=":",
                label="Optimal-split lower bound")
    ax.set_xlabel("Handover index")
    ax.set_ylabel("Cumulative latency (ms)")
    ax.set_title("Cumulative Latency: Bregman vs Random vs Greedy vs Oracle")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "7a_cumulative_latency_baselines.png"); plt.close(fig)

    # 7b: running-average latency ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(breg["ho_id"], breg["total_ms"].expanding().mean(),
            color=C_BREG, label="Bregman")
    ax.plot(rand["ho_id"], rand["total_ms"].expanding().mean(),
            color=C_RAND, linestyle="--", label="Random")
    if grdy is not None:
        ax.plot(grdy["ho_id"], grdy["total_ms"].expanding().mean(),
                color=C_GREEDY, linestyle="-.", label="Greedy full-info oracle")
    if "global_oracle_ms" in breg.columns:
        ax.plot(breg["ho_id"], breg["global_oracle_ms"].expanding().mean(),
                color=C_ORACLE, linestyle=":", label="Optimal-split lower bound")
    ax.set_xlabel("Handover index")
    ax.set_ylabel("Running-average total_ms (ms)")
    ax.set_title("Running-Average Latency Convergence")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out / "7b_running_avg_latency.png"); plt.close(fig)

    # 7c: gap-to-oracle over time ─────────────────────────────────────────────
    if "global_oracle_ms" in breg.columns and "global_oracle_ms" in rand.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        breg_gap = (breg["total_ms"] - breg["global_oracle_ms"]).clip(lower=0)
        rand_gap = (rand["total_ms"] - rand["global_oracle_ms"]).clip(lower=0)
        ax.plot(breg["ho_id"], breg_gap.expanding().mean(),
                color=C_BREG, label="Bregman gap-to-oracle (running avg)")
        ax.plot(rand["ho_id"], rand_gap.expanding().mean(),
                color=C_RAND, linestyle="--",
                label="Random gap-to-oracle (running avg)")
        if grdy is not None and "global_oracle_ms" in grdy.columns:
            grdy_gap = (grdy["total_ms"] - grdy["global_oracle_ms"]).clip(lower=0)
            ax.plot(grdy["ho_id"], grdy_gap.expanding().mean(),
                    color=C_GREEDY, linestyle="-.",
                    label="Greedy gap-to-oracle (running avg)")
        ax.set_xlabel("Handover index")
        ax.set_ylabel("Mean (total_ms − oracle) (ms)")
        ax.set_title("Convergence Gap to Optimal-Split Oracle")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.savefig(out / "7c_gap_to_oracle.png"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Selection Timeline Plots — Path and Instance Selection per Task Type
# ══════════════════════════════════════════════════════════════════════════════

def plot_selections(tau: str, ls: float, out: Path) -> None:
    """
    Create multi-panel figure showing path and instance selections with task types.
    Panels:
      1. Path selection (ISL vs GND) timeline with π_ISL overlay
      2. AMF instance selection per HO
      3. SMF instance selection per HO
      4. UPF instance selection per HO
    All with task-type background shading and labels.
    """
    df = _load(tau, ls, "dispatch_log")
    if df is None:
        return

    ho_ids = df["ho_id"].to_numpy()
    path = df["path"].to_numpy()
    p_isl = df["path_p_isl"].to_numpy()

    # Instance selection: extract which instance (0, 1, or 2) was chosen
    amf_inst = df["amf_inst"].to_numpy().astype(int)
    smf_inst = df["smf_inst"].to_numpy().astype(int)
    upf_inst = df["upf_inst"].to_numpy().astype(int)

    # Create 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Selection Timelines  (τ={tau}, load={ls})",
                 fontsize=14, fontweight="bold", y=0.995)

    # Panel 1: Path selection with π_ISL overlay
    ax = axes[0, 0]
    path_y = (path == "ISL").astype(int)
    ax.scatter(ho_ids, path_y, c=path_y, cmap="RdBu_r", s=20, alpha=0.6, zorder=3)
    ax2 = ax.twinx()
    ax2.plot(ho_ids, p_isl, color="#2196F3", lw=2, label="π_ISL", zorder=4)
    ax2.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.3)
    ax.set_ylabel("Path selected (1=ISL, 0=GND)")
    ax2.set_ylabel("π_ISL", color="#2196F3")
    ax2.tick_params(axis='y', labelcolor="#2196F3")
    ax2.legend(loc="upper left")
    ax.set_ylim(-0.1, 1.1)
    ax2.set_ylim(0, 1)
    ax.set_title("1. Path Selection Timeline")
    _add_task_background(ax, df)
    ax.grid(True, alpha=0.2)

    # Panel 2: AMF instance selection
    ax = axes[0, 1]
    colors_inst = ["#1565C0", "#EF6C00", "#2E7D32"]
    inst_names = ["ON-0", "ON-1/2", "GND"]
    for i in range(3):
        mask = amf_inst == i
        ax.scatter(ho_ids[mask], amf_inst[mask], c=colors_inst[i], s=30, alpha=0.7,
                   label=inst_names[i], zorder=3)
    ax.set_ylabel("AMF Instance")
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(inst_names)
    ax.set_title("2. AMF Instance Selection")
    _add_task_background(ax, df)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 3: SMF instance selection
    ax = axes[1, 0]
    for i in range(3):
        mask = smf_inst == i
        ax.scatter(ho_ids[mask], smf_inst[mask], c=colors_inst[i], s=30, alpha=0.7,
                   label=inst_names[i], zorder=3)
    ax.set_ylabel("SMF Instance")
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(inst_names)
    ax.set_xlabel("Handover index")
    ax.set_title("3. SMF Instance Selection")
    _add_task_background(ax, df)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel 4: UPF instance selection
    ax = axes[1, 1]
    for i in range(3):
        mask = upf_inst == i
        ax.scatter(ho_ids[mask], upf_inst[mask], c=colors_inst[i], s=30, alpha=0.7,
                   label=inst_names[i], zorder=3)
    ax.set_ylabel("UPF Instance")
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(inst_names)
    ax.set_xlabel("Handover index")
    ax.set_title("4. UPF Instance Selection")
    _add_task_background(ax, df)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fname = f"selections_{tau}_ls{_ls_tag(ls)}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    1: exp1_regret,
    2: exp2_probabilities,
    3: exp3_task_sensitivity,
    4: exp4_load_sweep,
    5: exp5_tau_sensitivity,
    6: exp6_geometry,
    7: exp7_baselines,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", type=int, nargs="*", default=list(EXPERIMENTS),
                   metavar="N", help="experiment numbers to run (default: all)")
    p.add_argument("--selections", action="store_true",
                   help="also generate selection timeline plots for all tau/load combos")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    for n in args.exp:
        if n not in EXPERIMENTS:
            print(f"[skip] unknown experiment {n}", file=sys.stderr)
            continue
        sub = OUT_DIR / f"exp{n}"
        sub.mkdir(exist_ok=True)
        print(f"\n══ Experiment {n} ══  → {sub.relative_to(HERE)}/")
        EXPERIMENTS[n](sub)

    # Generate selection timeline plots for all tau/load combinations
    if args.selections:
        print(f"\n══ Selection Timelines ══")
        selections_dir = OUT_DIR / "selections"
        selections_dir.mkdir(exist_ok=True)

        tau_profiles = ["strict", "default", "relaxed"]
        load_scales = [0.5, 1.0, 1.5, 2.0]

        for tau in tau_profiles:
            for ls in load_scales:
                print(f"  {tau} + load={ls}...", end=" ", flush=True)
                try:
                    plot_selections(tau, ls, selections_dir)
                except Exception as e:
                    print(f"error: {e}", file=sys.stderr)

    print(f"\nAll plots written under: {OUT_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
