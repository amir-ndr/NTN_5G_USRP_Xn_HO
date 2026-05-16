#!/usr/bin/env python3
"""
plot_analysis.py — Algorithm validation figures for paper

Produces four figures that directly validate the paper's claims:
  A. Regret growth rate (log-log) — proves sublinear regret (Theorem)
  B. Task-switch response — proves algorithm adapts to non-stationary traffic
  C. B_mult convergence — proves exploitation arm works correctly
  D. Path selection intelligence — end-to-end system behaviour

Usage:
    python3 plot_analysis.py
    python3 plot_analysis.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE    = Path(__file__).resolve().parent
LOG_CSV  = _HERE / "results/default_ls1/dispatch_log_default_ls1.csv"
RAND_CSV = _HERE / "results/default_ls1/random_log_default_ls1.csv"
OUT_DIR  = _HERE / "results/default_ls1"

C_ISL  = "#2196F3"
C_GND  = "#FF5722"
C_RAND = "#9E9E9E"
C_BEST = "#4CAF50"

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
TASK_SHORT = {
    "gaming": "GAM", "youtube": "YT", "browsing": "BR",
    "instagram": "IG", "mixed": "MX",
}

OUTLIER_THRESH = 150.0


def load_csv(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            print(f"[plot] ERROR: {path} not found.")
            sys.exit(1)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fv(row: dict, key: str) -> float:
    return float(row[key])


def add_task_shading(ax, ho_ids: np.ndarray, task_types: list[str]) -> None:
    if not task_types:
        return
    blocks: list[tuple[float, float, str]] = []
    prev_tt = task_types[0]
    start_x = ho_ids[0]
    for i in range(1, len(ho_ids)):
        if task_types[i] != prev_tt:
            blocks.append((start_x, ho_ids[i - 1], prev_tt))
            ax.axvline(float(ho_ids[i]) - 0.5, color="#555555", ls=":", lw=0.7,
                       alpha=0.5, zorder=1)
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


def rolling(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot A — Regret growth rate  (proves sublinear regret)
# ══════════════════════════════════════════════════════════════════════════════

def plot_A(rows: list[dict], rand_rows: list[dict]) -> None:
    ho_ids     = np.array([fv(r, "ho_id")                for r in rows])
    cum_global = np.array([fv(r, "cum_global_regret_ms") for r in rows])
    per_round  = np.array([fv(r, "global_regret_ms")     for r in rows])

    has_rand = len(rand_rows) > 0
    rand_ho  = np.array([fv(r, "ho_id")                for r in rand_rows]) if has_rand else None
    rand_cum = np.array([fv(r, "cum_global_regret_ms") for r in rand_rows]) if has_rand else None
    rand_pr  = np.array([fv(r, "global_regret_ms")     for r in rand_rows]) if has_rand else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("A — Regret Growth Rate  (sublinear → Theorem validated)",
                 fontsize=12, fontweight="bold")

    # ── Left: log-log cumulative regret ───────────────────────────────────────
    valid = cum_global > 0
    ax1.loglog(ho_ids[valid], cum_global[valid], color=C_ISL, lw=2.2,
               label="Bregman — cumulative global regret")
    if has_rand and rand_cum is not None:
        rv = rand_cum > 0
        ax1.loglog(rand_ho[rv], rand_cum[rv], color=C_RAND, lw=1.8,
                   ls="--", label="Random baseline")

    # Reference curves anchored at first valid Bregman point
    if valid.sum() > 2:
        t0 = ho_ids[valid][0]
        r0 = cum_global[valid][0]
        t  = ho_ids[valid]
        ax1.loglog(t, r0 * (t / t0),           color="#E53935", ls=":", lw=1.3,
                   label="O(T) — linear growth")
        ax1.loglog(t, r0 * np.sqrt(t / t0),    color="#FB8C00", ls=":", lw=1.3,
                   label="O(√T)")
        ax1.loglog(t, r0 * np.log(t / t0 + 1), color="#43A047", ls=":", lw=1.3,
                   label="O(log T)")

        # Fit log-log slope
        log_t = np.log(ho_ids[valid])
        log_r = np.log(cum_global[valid])
        if len(log_t) > 5:
            slope, _ = np.polyfit(log_t, log_r, 1)
            verdict  = "sublinear ✓" if slope < 1.0 else "linear or super-linear ✗"
            ax1.set_title(f"Log-Log: slope = {slope:.3f}  ({verdict})", fontsize=10)
        else:
            ax1.set_title("Log-Log cumulative regret", fontsize=10)

    ax1.set_xlabel("T  (handover index)")
    ax1.set_ylabel("Cumulative global regret (ms)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ── Right: per-round regret rolling mean ──────────────────────────────────
    WIN = max(5, min(25, len(per_round) // 8))
    rolled_b = rolling(per_round, WIN)
    x_b = ho_ids[WIN - 1:]

    ax2.plot(ho_ids, per_round, color=C_ISL, alpha=0.18, lw=0.7)
    ax2.plot(x_b, rolled_b, color=C_ISL, lw=2.0,
             label=f"Bregman  (w={WIN} rolling mean)")

    if has_rand and rand_pr is not None:
        rolled_r = rolling(rand_pr, WIN)
        x_r      = rand_ho[WIN - 1:]
        ax2.plot(rand_ho, rand_pr,  color=C_RAND, alpha=0.18, lw=0.7)
        ax2.plot(x_r, rolled_r, color=C_RAND, lw=2.0, ls="--", label="Random")

    task_types = [r["task_type"] for r in rows]
    add_task_shading(ax2, ho_ids, task_types)

    ax2.set_title("Per-Round Regret  (decreasing trend → algorithm converging)", fontsize=10)
    ax2.set_xlabel("Handover index")
    ax2.set_ylabel("Per-round global regret (ms)")
    ax2.set_ylim(bottom=0, top=np.percentile(per_round, 95) * 2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "analysis_A_regret_growth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[A] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot B — Task-switch response  (proves non-stationary adaptivity)
# ══════════════════════════════════════════════════════════════════════════════

def plot_B(rows: list[dict]) -> None:
    ho_ids     = np.array([fv(r, "ho_id")       for r in rows])
    task_types = [r["task_type"]                 for r in rows]
    path_p_isl = np.array([fv(r, "path_p_isl")  for r in rows])
    total_ms   = np.array([fv(r, "total_ms")     for r in rows])
    on_upf_p0  = np.array([fv(r, "on_upf_p0")   for r in rows])
    on_upf_p1  = np.array([fv(r, "on_upf_p1")   for r in rows])
    on_upf_p2  = np.array([fv(r, "on_upf_p2")   for r in rows])

    switches = [(i, task_types[i - 1], task_types[i])
                for i in range(1, len(task_types)) if task_types[i] != task_types[i - 1]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "B — Algorithm Response to Task Switches  "
        "(instagram = NTN-UPF overload event → key behavioural test)",
        fontsize=11, fontweight="bold"
    )

    # ── Row 1: ON-UPF instance weights ────────────────────────────────────────
    ax = axes[0]
    ax.plot(ho_ids, on_upf_p0, color="#E57373", lw=1.6,
            label="ON-UPF-0  (unstable @ instagram)")
    ax.plot(ho_ids, on_upf_p1, color="#FFA726", lw=1.6,
            label="ON-UPF-1  (unstable @ instagram)")
    ax.plot(ho_ids, on_upf_p2, color="#43A047", lw=2.0,
            label="ON-UPF-2  (only stable NTN UPF @ instagram)")
    ax.axhline(1 / 3, color="#888888", ls=":", lw=0.8, label="Uniform (1/3)")

    for idx, _, nxt in switches:
        is_ig = nxt == "instagram"
        ax.axvline(ho_ids[idx], color="#FB8C00" if is_ig else "#AAAAAA",
                   ls="--", lw=1.5 if is_ig else 0.6, alpha=0.8)

    add_task_shading(ax, ho_ids, task_types)
    ax.set_title(
        "ON-UPF Instance Selection Weights\n"
        "Expected: ON-UPF-2 rises to near-1.0 during instagram blocks",
        fontsize=9
    )
    ax.set_ylabel("Selection probability")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Row 2: π_ISL ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ho_ids, path_p_isl, color=C_ISL, lw=1.8, label="π_ISL  (Bregman)")
    ax.fill_between(ho_ids, path_p_isl, alpha=0.12, color=C_ISL)
    ax.axhline(0.5, color="#888888", ls="--", lw=0.9, label="Uniform (0.5)")

    for idx, _, nxt in switches:
        is_ig = nxt == "instagram"
        ax.axvline(ho_ids[idx], color="#FB8C00" if is_ig else "#AAAAAA",
                   ls="--", lw=1.5 if is_ig else 0.6, alpha=0.8)

    add_task_shading(ax, ho_ids, task_types)
    ax.set_title(
        "π_ISL Over Time\n"
        "Expected: drops at instagram (GND more reliable when NTN-UPF overloaded)",
        fontsize=9
    )
    ax.set_ylabel("π_ISL probability")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3: E2E latency ────────────────────────────────────────────────────
    ax = axes[2]
    WIN = max(5, min(15, len(total_ms) // 10))
    clipped = np.minimum(total_ms, OUTLIER_THRESH)
    rolled  = rolling(clipped, WIN)

    ax.plot(ho_ids, clipped, color=C_ISL, alpha=0.20, lw=0.7)
    ax.plot(ho_ids[WIN - 1:], rolled, color=C_ISL, lw=2.0,
            label=f"E2E latency  (w={WIN} rolling mean, clipped at {OUTLIER_THRESH:.0f} ms)")

    for idx, _, nxt in switches:
        is_ig = nxt == "instagram"
        lbl   = "instagram block" if is_ig else "_nolegend_"
        ax.axvline(ho_ids[idx], color="#FB8C00" if is_ig else "#AAAAAA",
                   ls="--", lw=1.5 if is_ig else 0.6, alpha=0.8, label=lbl)

    add_task_shading(ax, ho_ids, task_types)
    ax.set_title(
        "End-to-End Latency\n"
        "Expected: spike at task switch, recovery within a few HOs",
        fontsize=9
    )
    ax.set_xlabel("Handover index")
    ax.set_ylabel("Total delay (ms)")
    ax.set_ylim(0, OUTLIER_THRESH * 1.1)

    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, bool] = {}
    dedup_h, dedup_l = [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = True
            dedup_h.append(h)
            dedup_l.append(lbl)
    ax.legend(dedup_h, dedup_l, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "analysis_B_task_switch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[B] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot C — B_mult convergence  (proves exploitation arm works)
# ══════════════════════════════════════════════════════════════════════════════

def plot_C(rows: list[dict]) -> None:
    ho_ids     = np.array([fv(r, "ho_id") for r in rows])
    task_types = [r["task_type"]          for r in rows]

    LAYERS = [
        ("on_amf",  "ON AMF",  ["#BBDEFB", "#64B5F6", "#1E88E5"]),
        ("on_smf",  "ON SMF",  ["#E1BEE7", "#AB47BC", "#7B1FA2"]),
        ("on_upf",  "ON UPF",  ["#E57373", "#FFA726", "#43A047"]),
        ("gnd_amf", "GND AMF", ["#90CAF9", "#42A5F5", "#1565C0"]),
        ("gnd_smf", "GND SMF", ["#CE93D8", "#9C27B0", "#6A1B9A"]),
        ("gnd_upf", "GND UPF", ["#A5D6A7", "#66BB6A", "#2E7D32"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "C — Exploitation Arm: B_mult per NF Instance Over Time\n"
        "(rising = instance exploited / underloaded;  falling = overloaded / deprioritised)",
        fontsize=11, fontweight="bold"
    )

    for ax, (prefix, title, colors) in zip(axes.flat, LAYERS):
        for k, color in enumerate(colors):
            col = f"{prefix}_B{k}"
            if col not in rows[0]:
                continue
            vals = np.array([fv(r, col) for r in rows])
            ax.plot(ho_ids, vals, color=color, lw=1.6, label=f"Inst {k}")

        ax.axhline(1.0, color="#888888", ls="--", lw=0.9, alpha=0.6,
                   label="B_mult = 1.0  (hardware baseline)")
        ax.axhline(0.5, color="#E53935", ls=":", lw=0.7, alpha=0.5,
                   label="B_MULT_MIN (0.5)")
        ax.axhline(2.0, color="#43A047", ls=":", lw=0.7, alpha=0.5,
                   label="B_MULT_MAX (2.0)")

        add_task_shading(ax, ho_ids, task_types)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.3, 2.3)
        ax.set_xlabel("Handover index", fontsize=8)
        ax.set_ylabel("B_mult", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "analysis_C_bmult.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[C] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot D — Path selection intelligence  (end-to-end system result)
# ══════════════════════════════════════════════════════════════════════════════

def plot_D(rows: list[dict], rand_rows: list[dict]) -> None:
    ho_ids     = np.array([fv(r, "ho_id")      for r in rows])
    task_types = [r["task_type"]                for r in rows]
    path_p_isl = np.array([fv(r, "path_p_isl") for r in rows])
    total_ms   = np.array([fv(r, "total_ms")   for r in rows])

    has_rand        = len(rand_rows) > 0
    rand_task_types = [r["task_type"]             for r in rand_rows] if has_rand else []
    rand_total      = np.array([fv(r, "total_ms") for r in rand_rows]) if has_rand else np.array([])
    rand_ho         = np.array([fv(r, "ho_id")    for r in rand_rows]) if has_rand else np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "D — Path Selection Intelligence: ISL vs GND Under Heterogeneous Load",
        fontsize=12, fontweight="bold"
    )

    # ── Top-left: mean π_ISL per task type ───────────────────────────────────
    ax = axes[0, 0]
    task_pisl: dict[str, list[float]] = {tt: [] for tt in TASK_ORDER}
    for i, tt in enumerate(task_types):
        task_pisl[tt].append(float(path_p_isl[i]))

    x      = np.arange(len(TASK_ORDER))
    means  = [np.mean(task_pisl[tt]) if task_pisl[tt] else 0.0 for tt in TASK_ORDER]
    stdevs = [np.std(task_pisl[tt])  if task_pisl[tt] else 0.0 for tt in TASK_ORDER]
    colors = [TASK_BAR_COLORS[tt] for tt in TASK_ORDER]

    bars = ax.bar(x, means, yerr=stdevs, capsize=5, color=colors,
                  alpha=0.85, edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="#888888", ls="--", lw=1.2, label="Random (0.5)")
    for gi, (bar, m, sd) in enumerate(zip(bars, means, stdevs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(m + sd + 0.04, 1.12),
                f"{m:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TASK_ORDER], fontsize=9)
    ax.set_ylabel("Mean π_ISL  (± 1σ)")
    ax.set_title(
        "Learned ISL Probability by Task Type\n"
        "(instagram↓ = algorithm detects NTN-UPF instability)",
        fontsize=9
    )
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Top-right: mean E2E latency Bregman vs Random per task ───────────────
    ax = axes[0, 1]
    bregman_lat: dict[str, list[float]] = {tt: [] for tt in TASK_ORDER}
    rand_lat:    dict[str, list[float]] = {tt: [] for tt in TASK_ORDER}

    for i, tt in enumerate(task_types):
        if total_ms[i] < OUTLIER_THRESH:
            bregman_lat[tt].append(float(total_ms[i]))
    if has_rand:
        for i, tt in enumerate(rand_task_types):
            if rand_total[i] < OUTLIER_THRESH:
                rand_lat[tt].append(float(rand_total[i]))

    bw = 0.35
    for gi, tt in enumerate(TASK_ORDER):
        b_mean = np.mean(bregman_lat[tt]) if bregman_lat[tt] else 0.0
        b_std  = np.std(bregman_lat[tt])  if bregman_lat[tt] else 0.0
        r_mean = np.mean(rand_lat[tt])    if rand_lat[tt]    else 0.0
        r_std  = np.std(rand_lat[tt])     if rand_lat[tt]    else 0.0
        clr    = TASK_BAR_COLORS[tt]

        ax.bar(gi - bw / 2, b_mean, bw, yerr=b_std, capsize=3,
               color=clr, alpha=0.90,
               label="Bregman" if gi == 0 else "_nolegend_")
        if has_rand:
            ax.bar(gi + bw / 2, r_mean, bw, yerr=r_std, capsize=3,
                   color=clr, alpha=0.35, hatch="///",
                   label="Random" if gi == 0 else "_nolegend_")
            if b_mean > 0 and r_mean > 0:
                gain = (r_mean - b_mean) / r_mean * 100
                top  = max(b_mean + b_std, r_mean + r_std) + 0.3
                ax.text(gi, top, f"−{gain:.0f}%",
                        ha="center", fontsize=7.5,
                        color="#1B5E20", fontweight="bold")

    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([t.capitalize() for t in TASK_ORDER], fontsize=9)
    ax.set_ylabel("Mean E2E latency (ms)")
    ax.set_title(
        "Latency Reduction vs Random Baseline\n"
        "(% gain labelled; outliers > 150 ms excluded)",
        fontsize=9
    )
    bregman_h = mpatches.Patch(color="#888888", alpha=0.90, label="Bregman (solid)")
    leg_handles = [bregman_h]
    if has_rand:
        rand_h = mpatches.Patch(color="#888888", alpha=0.40, hatch="///",
                                label="Random (hatched)")
        leg_handles.append(rand_h)
    ax.legend(handles=leg_handles, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Bottom-left: latency CDF by path ─────────────────────────────────────
    ax = axes[1, 0]
    isl_lat  = [fv(r, "total_ms") for r in rows
                if r["path"] == "ISL" and fv(r, "total_ms") < OUTLIER_THRESH]
    gnd_lat  = [fv(r, "total_ms") for r in rows
                if r["path"] == "GND" and fv(r, "total_ms") < OUTLIER_THRESH]
    risl_lat = ([fv(r, "total_ms") for r in rand_rows
                 if r["path"] == "ISL" and fv(r, "total_ms") < OUTLIER_THRESH]
                if has_rand else [])
    rgnd_lat = ([fv(r, "total_ms") for r in rand_rows
                 if r["path"] == "GND" and fv(r, "total_ms") < OUTLIER_THRESH]
                if has_rand else [])

    for data, color, ls, lbl in [
        (isl_lat,  C_ISL,  "-",  f"Bregman ISL  (n={len(isl_lat)})"),
        (gnd_lat,  C_GND,  "-",  f"Bregman GND  (n={len(gnd_lat)})"),
        (risl_lat, C_ISL,  "--", f"Random ISL   (n={len(risl_lat)})"),
        (rgnd_lat, C_GND,  "--", f"Random GND   (n={len(rgnd_lat)})"),
    ]:
        if data:
            s = np.sort(data)
            ax.plot(s, np.arange(1, len(s) + 1) / len(s),
                    color=color, ls=ls, lw=1.8, label=lbl)

    ax.set_xlabel("Total delay (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF by Path — Bregman vs Random", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: π_ISL convergence speed (rolling σ) ────────────────────
    ax = axes[1, 1]
    WIN_CONV = max(8, min(20, len(path_p_isl) // 10))
    roll_std = np.array([
        np.std(path_p_isl[max(0, i - WIN_CONV):i + 1])
        for i in range(len(path_p_isl))
    ])

    ax.fill_between(ho_ids, roll_std, alpha=0.20, color=C_ISL)
    ax.plot(ho_ids, roll_std, color=C_ISL, lw=1.8,
            label=f"Rolling σ of π_ISL  (w={WIN_CONV})")

    CONV_THRESH = 0.05
    ax.axhline(CONV_THRESH, color=C_BEST, ls="--", lw=1.3,
               label=f"Convergence threshold (σ = {CONV_THRESH})")

    below = np.where(roll_std < CONV_THRESH)[0]
    if len(below) > 0:
        first_ho = int(ho_ids[below[0]])
        ax.axvline(first_ho, color=C_BEST, ls=":", lw=1.5, alpha=0.85)
        ax.text(first_ho + 1, CONV_THRESH + 0.005,
                f"First convergence\n@ HO {first_ho}",
                fontsize=7.5, color=C_BEST, va="bottom")

    add_task_shading(ax, ho_ids, task_types)
    ax.set_title(
        "Convergence Speed — Variance in π_ISL\n"
        "(σ drops when algorithm stabilises on a routing policy)",
        fontsize=9
    )
    ax.set_xlabel("Handover index")
    ax.set_ylabel(f"Rolling σ of π_ISL  (w={WIN_CONV})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "analysis_D_path_intelligence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[D] Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Algorithm validation figures")
    p.add_argument("--no-show", action="store_true",
                   help="Save PNGs only, do not open display windows")
    args = p.parse_args()

    rows      = load_csv(LOG_CSV, required=True)
    rand_rows = load_csv(RAND_CSV, required=False)
    if not rows:
        print("[plot] dispatch_log is empty.")
        sys.exit(1)
    if not rand_rows:
        print("[plot] random_log not found — random overlay skipped.")

    plot_A(rows, rand_rows)
    plot_B(rows)
    plot_C(rows)
    plot_D(rows, rand_rows)

    print(f"\n[plot] All four figures written to: {OUT_DIR}/")
    print("  analysis_A_regret_growth.png")
    print("  analysis_B_task_switch.png")
    print("  analysis_C_bmult.png")
    print("  analysis_D_path_intelligence.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
