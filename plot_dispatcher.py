#!/usr/bin/env python3
"""
plot_dispatcher.py — Visualise dispatcher learning results

8-panel figure (4×2 grid):
  1. PathScheduler — learned π_ISL / π_GND  (Level-1 Bregman probabilities only)
  2. Access cost signals — normalised cost_ISL vs cost_GND  (driving signal for Level-1)
  3. End-to-end delay per HO — outlier-corrected running average
  4. Per-task cumulative regret — 5 lines, one per task type
  5. Per-task-type mean latency — Bregman vs Random (non-outlier HOs only)
  6. Cumulative regret decomposition — access + NF instance + global
  7. Ground Core — per-layer instance selection probabilities
  8. Onboard Core — per-layer instance selection probabilities

Usage:
    python3 plot_dispatcher.py
    python3 plot_dispatcher.py --no-show     # save PNG only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE    = Path(__file__).resolve().parent
LOG_CSV  = _HERE / "dispatch_log.csv"
RAND_CSV = _HERE / "random_log.csv"
OUT_PNG  = _HERE / "results_dispatcher.png"

CLIP_MS        = 70.0   # display clip for delay panel
OUTLIER_THRESH = 150.0  # exclude from per-task averages

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
    """Draw semi-transparent task-type bands + vertical boundary lines."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    rows = load_log(LOG_CSV, required=True)
    if not rows:
        print("[plot] dispatch_log.csv is empty.")
        sys.exit(1)

    rand_rows = load_log(RAND_CSV, required=False)
    has_rand  = len(rand_rows) > 0
    if not has_rand:
        print("[plot] random_log.csv not found — random baseline overlay skipped.")

    n      = len(rows)
    ho_ids = np.array([fv(r, "ho_id") for r in rows])

    colours  = [C_ISL if r["path"] == "ISL" else C_GND for r in rows]

    prop_ms  = np.array([fv(r, "prop_ms")          for r in rows])
    amf_ms   = np.array([fv(r, "amf_ms")           for r in rows])
    smf_ms   = np.array([fv(r, "smf_ms")           for r in rows])
    upf_ms   = np.array([fv(r, "upf_ms")           for r in rows])
    total_ms = np.array([fv(r, "total_ms")          for r in rows])
    oracle   = np.array([fv(r, "global_oracle_ms")  for r in rows])

    path_p_isl   = np.array([fv(r, "path_p_isl")      for r in rows])
    path_p_gnd   = np.array([fv(r, "path_p_gnd")      for r in rows])
    acc_cost_isl = np.array([fv(r, "access_cost_isl") for r in rows])
    acc_cost_gnd = np.array([fv(r, "access_cost_gnd") for r in rows])
    cum_access_reg = np.array([fv(r, "cum_access_regret_ms") for r in rows])
    cum_inst_reg   = np.array([fv(r, "cum_inst_regret_ms")   for r in rows])
    cum_global_reg = np.array([fv(r, "cum_global_regret_ms") for r in rows])
    global_reg_ms  = np.array([fv(r, "global_regret_ms")     for r in rows])

    isl_mask   = np.array([r["path"] == "ISL" for r in rows])
    gnd_mask   = ~isl_mask
    task_types = [r["task_type"] for r in rows]

    def layer_probs(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([fv(r, f"{prefix}_p0") for r in rows]),
            np.array([fv(r, f"{prefix}_p1") for r in rows]),
            np.array([fv(r, f"{prefix}_p2") for r in rows]),
        )

    gnd_amf_p = layer_probs("gnd_amf")
    gnd_smf_p = layer_probs("gnd_smf")
    gnd_upf_p = layer_probs("gnd_upf")
    on_amf_p  = layer_probs("on_amf")
    on_smf_p  = layer_probs("on_smf")
    on_upf_p  = layer_probs("on_upf")

    rand_ho_ids   = np.array([])
    rand_total    = np.array([])
    rand_cum_reg  = np.array([])
    rand_isl_mask = np.array([], dtype=bool)
    rand_gnd_mask = np.array([], dtype=bool)
    rand_task_types: list[str] = []
    nr = 0

    if has_rand:
        nr              = len(rand_rows)
        rand_ho_ids     = np.array([fv(r, "ho_id")         for r in rand_rows])
        rand_colours    = [C_ISL if r["path"] == "ISL" else C_GND for r in rand_rows]
        rand_total      = np.array([fv(r, "total_ms")      for r in rand_rows])
        rand_cum_reg         = np.array([fv(r, "cum_global_regret_ms") for r in rand_rows])
        rand_cum_access_reg  = np.array([fv(r, "cum_access_regret_ms") for r in rand_rows])
        rand_cum_inst_reg    = np.array([fv(r, "cum_inst_regret_ms")   for r in rand_rows])
        rand_isl_mask   = np.array([r["path"] == "ISL"     for r in rand_rows])
        rand_gnd_mask   = ~rand_isl_mask
        rand_task_types = [r["task_type"] for r in rand_rows]

    # ── Per-task cumulative regret (panel 4) ──────────────────────────────────
    # For each task type, collect global_regret_ms in HO order, compute cumsum.
    task_reg: dict[str, list[tuple[int, float]]] = {tt: [] for tt in TASK_ORDER}
    for i, tt in enumerate(task_types):
        task_reg[tt].append((int(ho_ids[i]), float(global_reg_ms[i])))

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle("NTN Dispatcher — Online Learning Results  (Bregman vs Random Baseline)",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(5, 2, hspace=0.46, wspace=0.32)

    ax_prob   = fig.add_subplot(gs[0, 0])   # 1. PathScheduler probs
    ax_cost   = fig.add_subplot(gs[0, 1])   # 2. Access cost signals
    ax_del    = fig.add_subplot(gs[1, 0])   # 3. End-to-end delay
    ax_treg   = fig.add_subplot(gs[1, 1])   # 4. Per-task cumulative regret
    ax_task   = fig.add_subplot(gs[2, 0])   # 5. Per-task mean latency
    ax_reg    = fig.add_subplot(gs[2, 1])   # 6. Cumulative regret decomposition
    ax_gnd    = fig.add_subplot(gs[3, 0])   # 7. Ground core probs
    ax_on     = fig.add_subplot(gs[3, 1])   # 8. Onboard core probs
    ax_lb     = fig.add_subplot(gs[4, :])   # 9. Load balancing (full width)

    # ── 1. PathScheduler — learned π_ISL / π_GND ─────────────────────────────
    ax_prob.set_title("PathScheduler — Learned π_ISL / π_GND  (Level-1 Bregman)")
    ax_prob.set_xlabel("Handover index")
    ax_prob.set_ylabel("Path probability")

    ax_prob.fill_between(ho_ids, path_p_isl, alpha=0.12, color=C_ISL)
    ax_prob.fill_between(ho_ids, path_p_gnd, alpha=0.12, color=C_GND)
    ax_prob.plot(ho_ids, path_p_isl, color=C_ISL, lw=1.8, label="π_ISL (learned)")
    ax_prob.plot(ho_ids, path_p_gnd, color=C_GND, lw=1.8, label="π_GND (learned)")
    ax_prob.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="uniform (random)")

    add_task_shading(ax_prob, ho_ids, task_types)
    ax_prob.set_ylim(0, 1.12)
    ax_prob.legend(fontsize=8, loc="upper right")
    ax_prob.grid(True, alpha=0.3)

    # ── 2. Access cost signals — driving signal for PathScheduler ─────────────
    ax_cost.set_title("Access Cost Signals — ISL vs GND  (Level-1 Bregman input)")
    ax_cost.set_xlabel("Handover index")
    ax_cost.set_ylabel("Access cost (ms)")

    ax_cost.plot(ho_ids, acc_cost_isl, color=C_ISL, lw=0.9, alpha=0.7, label="cost_ISL (ms)")
    ax_cost.plot(ho_ids, acc_cost_gnd, color=C_GND, lw=0.9, alpha=0.7, label="cost_GND (ms)")

    # Rolling mean for readability
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

    # ── 3. Total delay per HO ─────────────────────────────────────────────────
    total_ms_clipped = np.minimum(total_ms, CLIP_MS)
    is_clipped       = total_ms > CLIP_MS
    n_clipped        = int(is_clipped.sum())

    ax_del.set_title(
        "End-to-End Delay per Handover"
        + (f"  ({n_clipped} outliers >{CLIP_MS:.0f} ms)" if n_clipped else "")
    )
    ax_del.set_xlabel("Handover index")
    ax_del.set_ylabel("Total delay (ms)")

    if has_rand:
        rand_total_clipped = np.minimum(rand_total, CLIP_MS)
        rand_is_clipped    = rand_total > CLIP_MS
        ax_del.scatter(rand_ho_ids, rand_total_clipped,
                       c=C_RAND, s=9, alpha=0.28, zorder=2, label="_nolegend_")
        if rand_is_clipped.any():
            ax_del.scatter(rand_ho_ids[rand_is_clipped],
                           np.full(rand_is_clipped.sum(), CLIP_MS),
                           marker="^", c=C_RAND, s=16, alpha=0.40, zorder=2)
        WIN_R = max(1, min(20, nr // 4))
        rx_idx, ry = rolling_mean_valid(rand_total, CLIP_MS, WIN_R)
        if len(rx_idx) > 1:
            ax_del.plot(rand_ho_ids[rx_idx], ry, color=C_RAND2, lw=1.5,
                        ls="--", label=f"Random avg (valid,w={WIN_R})", zorder=3)

    ax_del.scatter(ho_ids[~is_clipped], total_ms_clipped[~is_clipped],
                   c=np.array(colours)[~is_clipped], s=14, alpha=0.70, zorder=4)
    if is_clipped.any():
        ax_del.scatter(ho_ids[is_clipped],
                       np.full(is_clipped.sum(), CLIP_MS),
                       marker="^", c=np.array(colours)[is_clipped],
                       s=28, alpha=0.80, zorder=4,
                       label=f"Clipped (>{CLIP_MS:.0f} ms)")

    oracle_clipped = np.minimum(oracle, CLIP_MS)
    ax_del.plot(ho_ids, oracle_clipped, color=C_BEST, lw=1.2, ls="--",
                label="Global oracle", zorder=5)

    WIN_B = max(1, min(20, n // 4))
    bx_idx, by = rolling_mean_valid(total_ms, CLIP_MS, WIN_B)
    if len(bx_idx) > 1:
        ax_del.plot(ho_ids[bx_idx], by, color="black", lw=1.6,
                    label=f"Bregman avg (valid,w={WIN_B})", zorder=5)

    add_task_shading(ax_del, ho_ids, task_types)

    legend_handles = list(ax_del.get_legend_handles_labels()[0]) + [
        mpatches.Patch(color=C_ISL, label="ISL (Bregman)"),
        mpatches.Patch(color=C_GND, label="GND (Bregman)"),
    ]
    if has_rand:
        legend_handles.append(mpatches.Patch(color=C_RAND, label="Random"))
    ax_del.legend(fontsize=7.5, handles=legend_handles)
    ax_del.grid(True, alpha=0.3)

    # ── 4. Per-task cumulative regret ─────────────────────────────────────────
    ax_treg.set_title("Per-Task Cumulative Regret  (global regret by task type)")
    ax_treg.set_xlabel("Per-task handover index")
    ax_treg.set_ylabel("Cumulative regret (ms)")

    for tt in TASK_ORDER:
        entries = task_reg[tt]
        if not entries:
            continue
        reg_vals = np.array([v for _, v in entries])
        cum_r    = np.cumsum(reg_vals)
        x_idx    = np.arange(1, len(cum_r) + 1)
        ax_treg.plot(x_idx, cum_r, color=TASK_LINE_COLORS[tt],
                     lw=1.6, label=f"{tt} (n={len(cum_r)})")

    ax_treg.legend(fontsize=8, ncol=1, loc="upper left")
    ax_treg.grid(True, alpha=0.3)
    ax_treg.annotate("x-axis = per-task HO count (not global)",
                     xy=(0.99, 0.02), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=6, color="gray")

    # ── 5. Per-task-type mean latency ─────────────────────────────────────────
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
            task_bregman[tt]["prop_ms"].append(float(prop_ms[i]))
            task_bregman[tt]["amf_ms"].append(float(amf_ms[i]))
            task_bregman[tt]["smf_ms"].append(float(smf_ms[i]))
            task_bregman[tt]["upf_ms"].append(float(upf_ms[i]))

    task_rand: dict[str, dict[str, list[float]]] = {
        tt: {k: [] for k in comp_keys} for tt in TASK_ORDER
    }
    if has_rand:
        rand_prop = np.array([fv(r, "prop_ms") for r in rand_rows])
        rand_amf  = np.array([fv(r, "amf_ms")  for r in rand_rows])
        rand_smf  = np.array([fv(r, "smf_ms")  for r in rand_rows])
        rand_upf  = np.array([fv(r, "upf_ms")  for r in rand_rows])
        for i, tt in enumerate(rand_task_types):
            if rand_total[i] < OUTLIER_THRESH:
                task_rand[tt]["prop_ms"].append(float(rand_prop[i]))
                task_rand[tt]["amf_ms"].append(float(rand_amf[i]))
                task_rand[tt]["smf_ms"].append(float(rand_smf[i]))
                task_rand[tt]["upf_ms"].append(float(rand_upf[i]))

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

        b_bottom = 0.0
        r_bottom = 0.0
        for key, lbl, clr, al in COMP_SPECS:
            b_mean = float(np.mean(task_bregman[tt][key])) if task_bregman[tt][key] else 0.0
            ax_task.bar(x_task[gi] - bar_w / 2, b_mean, bar_w,
                        bottom=b_bottom, color=clr, alpha=al,
                        label=lbl if gi == 0 else "_nolegend_", edgecolor="white", lw=0.4)
            if b_mean > 0.5:
                ax_task.text(x_task[gi] - bar_w / 2, b_bottom + b_mean / 2,
                             f"{b_mean:.1f}", ha="center", va="center",
                             fontsize=5.5, color="white", fontweight="bold")
            b_bottom += b_mean

            if has_rand:
                r_mean = float(np.mean(task_rand[tt][key])) if task_rand[tt][key] else 0.0
                ax_task.bar(x_task[gi] + bar_w / 2, r_mean, bar_w,
                            bottom=r_bottom, color=clr, alpha=al * 0.45,
                            hatch="///", label="_nolegend_", edgecolor="white", lw=0.4)
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
        rand_h = mpatches.Patch(color="#888888", alpha=0.38, hatch="///", label="Random (hatched)")
        handles.append(rand_h)
    ax_task.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")
    ax_task.grid(axis="y", alpha=0.3)
    ax_task.annotate(f"Outliers >{OUTLIER_THRESH:.0f} ms excluded",
                     xy=(0.01, 0.98), xycoords="axes fraction",
                     va="top", fontsize=6.5, color="gray")

    # ── 6. Cumulative regret — decomposed ────────────────────────────────────
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

    # ── 7. Ground core — per-layer instance probabilities ────────────────────
    ax_gnd.set_title("Ground Core — Instance Selection Probabilities")
    ax_gnd.set_xlabel("Handover index")
    ax_gnd.set_ylabel("Selection probability")

    styles = ["-", "--", ":"]
    for i, (prob, sty) in enumerate(zip(gnd_amf_p, styles)):
        ax_gnd.plot(ho_ids, prob, color="#1565C0", ls=sty, lw=1.2, label=f"AMF-GND-{i}")
    for i, (prob, sty) in enumerate(zip(gnd_smf_p, styles)):
        ax_gnd.plot(ho_ids, prob, color="#FF5722", ls=sty, lw=1.2, label=f"SMF-GND-{i}")
    for i, (prob, sty) in enumerate(zip(gnd_upf_p, styles)):
        ax_gnd.plot(ho_ids, prob, color="#4CAF50", ls=sty, lw=1.2, label=f"UPF-GND-{i}")

    add_task_shading(ax_gnd, ho_ids, task_types)
    ax_gnd.axhline(1 / 3, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax_gnd.set_ylim(0, 1.10)
    ax_gnd.legend(fontsize=7, ncol=3)
    ax_gnd.grid(True, alpha=0.3)
    ax_gnd.annotate("UPF probs reflect current task type's per-task scheduler",
                    xy=(0.01, 0.98), xycoords="axes fraction",
                    va="top", fontsize=6, color="gray")

    # ── 8. Onboard core — per-layer instance probabilities ───────────────────
    ax_on.set_title("Onboard Core — Instance Selection Probabilities")
    ax_on.set_xlabel("Handover index")
    ax_on.set_ylabel("Selection probability")

    for i, (prob, sty) in enumerate(zip(on_amf_p, styles)):
        ax_on.plot(ho_ids, prob, color="#1565C0", ls=sty, lw=1.2, label=f"AMF-ON-{i}")
    for i, (prob, sty) in enumerate(zip(on_smf_p, styles)):
        ax_on.plot(ho_ids, prob, color="#FF5722", ls=sty, lw=1.2, label=f"SMF-ON-{i}")
    for i, (prob, sty) in enumerate(zip(on_upf_p, styles)):
        ax_on.plot(ho_ids, prob, color="#4CAF50", ls=sty, lw=1.2, label=f"UPF-ON-{i}")

    add_task_shading(ax_on, ho_ids, task_types)
    ax_on.axhline(1 / 3, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax_on.set_ylim(0, 1.10)
    ax_on.legend(fontsize=7, ncol=3)
    ax_on.grid(True, alpha=0.3)
    ax_on.annotate("UPF probs reflect current task type's per-task scheduler",
                   xy=(0.01, 0.98), xycoords="axes fraction",
                   va="top", fontsize=6, color="gray")

    # ── 9. Load balancing — bg_load vs learned routing probabilities (full width) ──
    ax_lb.set_title(
        "Access Node Load Balancing — bg_load vs Learned π_ISL  "
        "(ISL wins when bg_load_ISL < ~0.67; tradeoff is xn_setup, not propagation)"
    )
    ax_lb.set_xlabel("Handover index")
    ax_lb.set_ylabel("Probability / bg_load  [0 – 1]")

    trgsat_bg_arr = np.array([fv(r, "trgsat_bg") for r in rows])
    tn_bg_arr     = np.array([fv(r, "tn_bg")     for r in rows])

    # inv_cost_p_isl was added in a later schema version; fall back to computing
    # it from the always-present access_cost_isl / access_cost_gnd columns.
    if "inv_cost_p_isl" in rows[0]:
        inv_p_isl = np.array([fv(r, "inv_cost_p_isl") for r in rows])
    else:
        c_isl = np.array([fv(r, "access_cost_isl") for r in rows])
        c_gnd = np.array([fv(r, "access_cost_gnd") for r in rows])
        w_isl = 1.0 / np.maximum(c_isl, 0.01)
        w_gnd = 1.0 / np.maximum(c_gnd, 0.01)
        inv_p_isl = w_isl / (w_isl + w_gnd)

    # Background load fills — show congestion level of each node over time
    ax_lb.fill_between(ho_ids, trgsat_bg_arr, alpha=0.18, color=C_ISL,
                       label="bg_load ISL (TrgSAT)")
    ax_lb.fill_between(ho_ids, tn_bg_arr,     alpha=0.18, color=C_GND,
                       label="bg_load GND (TN)")
    ax_lb.plot(ho_ids, trgsat_bg_arr, color=C_ISL, lw=0.8, alpha=0.5)
    ax_lb.plot(ho_ids, tn_bg_arr,     color=C_GND, lw=0.8, alpha=0.5)

    # Crossover reference: ISL/GND costs are equal around bg_load_ISL ≈ 0.67
    ax_lb.axhline(0.67, color="#888888", ls=":", lw=1.0, alpha=0.6,
                  label="bg_ISL ≈ 0.67  (ISL↔GND cost crossover)")

    # Inverse-cost target: what a perfect instantaneous load-balancer routes to ISL
    win_lb = max(1, min(20, n // 6))
    _, ic_rm = rolling_mean_valid(inv_p_isl, 1e9, win_lb)
    x_lb = np.arange(len(ic_rm))
    if len(x_lb) > 1:
        ax_lb.plot(ho_ids[x_lb], ic_rm, color="black", lw=1.4, ls="--",
                   label=f"Inv-cost π_ISL (greedy optimum, avg w={win_lb})")

    # Bregman learned probability
    ax_lb.plot(ho_ids, path_p_isl, color=C_ISL, lw=2.0,
               label="π_ISL  (Bregman learned)")

    add_task_shading(ax_lb, ho_ids, task_types)
    ax_lb.set_ylim(0, 1.05)
    ax_lb.legend(fontsize=8, ncol=3, loc="upper right")
    ax_lb.grid(True, alpha=0.3)
    ax_lb.annotate(
        "When bg_load_ISL (blue fill) rises above ~0.67, "
        "greedy optimum (dashed) drops → Bregman π_ISL should follow with a lag",
        xy=(0.01, 0.03), xycoords="axes fraction",
        va="bottom", fontsize=7, color="#444444",
    )

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
    print(f"  Per-task cumulative regret (global):")
    for tt in TASK_ORDER:
        entries = task_reg[tt]
        if entries:
            cum = sum(v for _, v in entries)
            avg = cum / len(entries)
            print(f"    {tt:<12}: cum={cum:.1f} ms  avg/HO={avg:.2f} ms  n={len(entries)}")

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

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
