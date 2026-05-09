#!/usr/bin/env python3
"""
plot_dispatcher.py — Visualise dispatcher learning results

Reads dispatch_log.csv (Bregman algorithm) and optionally random_log.csv
(uniform random baseline) and produces a 6-panel figure:

  1. Path selection weights over HO index       — Bregman convergence + random 0.5 ref
  2. Total end-to-end delay per HO              — scatter coloured by path, both algos
  3. Latency breakdown (stacked bars)           — prop + AMF + SMF + UPF, both algos
  4. Per-layer instance probabilities (ground)  — AMF / SMF / UPF convergence
  5. Per-layer instance probabilities (onboard) — same for ISL path
  6. Cumulative regret vs HO index              — Bregman vs random

Usage:
    python3 plot_dispatcher.py
    python3 plot_dispatcher.py --no-show     # save PNG only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE    = Path(__file__).resolve().parent
LOG_CSV  = _HERE / "dispatch_log.csv"
RAND_CSV = _HERE / "random_log.csv"
OUT_PNG  = _HERE / "results_dispatcher.png"

C_ISL   = "#2196F3"   # blue
C_GND   = "#FF5722"   # orange
C_BEST  = "#4CAF50"   # green  (oracle)
C_RAND  = "#9E9E9E"   # gray   (random baseline)
C_RAND2 = "#757575"   # darker gray (random running avg / regret)


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


def main():
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
    ho_ids = [fv(r, "ho_id") for r in rows]

    colours  = [C_ISL if r["path"] == "ISL" else C_GND for r in rows]

    prop_ms  = np.array([fv(r, "prop_ms")       for r in rows])
    amf_ms   = np.array([fv(r, "amf_ms")        for r in rows])
    smf_ms   = np.array([fv(r, "smf_ms")        for r in rows])
    upf_ms   = np.array([fv(r, "upf_ms")        for r in rows])
    total_ms = np.array([fv(r, "total_ms")      for r in rows])
    oracle   = np.array([fv(r, "oracle_ms")     for r in rows])
    cum_reg  = np.array([fv(r, "cum_regret_ms") for r in rows])

    p_isl = np.array([fv(r, "p_isl") for r in rows])
    p_gnd = np.array([fv(r, "p_gnd") for r in rows])

    isl_mask = np.array([r["path"] == "ISL" for r in rows])
    gnd_mask = ~isl_mask

    def layer_probs(prefix):
        p0 = np.array([fv(r, f"{prefix}_p0") for r in rows])
        p1 = np.array([fv(r, f"{prefix}_p1") for r in rows])
        p2 = np.array([fv(r, f"{prefix}_p2") for r in rows])
        return p0, p1, p2

    gnd_amf_p = layer_probs("gnd_amf")
    gnd_smf_p = layer_probs("gnd_smf")
    gnd_upf_p = layer_probs("gnd_upf")
    on_amf_p  = layer_probs("on_amf")
    on_smf_p  = layer_probs("on_smf")
    on_upf_p  = layer_probs("on_upf")

    # Random baseline arrays
    if has_rand:
        nr           = len(rand_rows)
        rand_ho_ids  = [fv(r, "ho_id")         for r in rand_rows]
        rand_colours = [C_ISL if r["path"] == "ISL" else C_GND for r in rand_rows]
        rand_prop    = np.array([fv(r, "prop_ms")       for r in rand_rows])
        rand_amf     = np.array([fv(r, "amf_ms")        for r in rand_rows])
        rand_smf     = np.array([fv(r, "smf_ms")        for r in rand_rows])
        rand_upf     = np.array([fv(r, "upf_ms")        for r in rand_rows])
        rand_total   = np.array([fv(r, "total_ms")      for r in rand_rows])
        rand_cum_reg = np.array([fv(r, "cum_regret_ms") for r in rand_rows])
        rand_isl_mask = np.array([r["path"] == "ISL" for r in rand_rows])
        rand_gnd_mask = ~rand_isl_mask

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("NTN Dispatcher — Online Learning Results (Bregman vs Random Baseline)",
                 fontsize=13, fontweight="bold")
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30)

    ax_path = fig.add_subplot(gs[0, 0])
    ax_del  = fig.add_subplot(gs[0, 1])
    ax_bar  = fig.add_subplot(gs[1, 0])
    ax_gnd  = fig.add_subplot(gs[1, 1])
    ax_on   = fig.add_subplot(gs[2, 0])
    ax_reg  = fig.add_subplot(gs[2, 1])

    # ── 1. Path-selection weights ─────────────────────────────────────────────
    ax_path.set_title("Dispatcher Path Weights (Hedge)")
    ax_path.set_xlabel("Handover index")
    ax_path.set_ylabel("Selection probability")
    ax_path.plot(ho_ids, p_isl, color=C_ISL, lw=1.5, label="Bregman P(ISL)")
    ax_path.plot(ho_ids, p_gnd, color=C_GND, lw=1.5, label="Bregman P(GND)")
    ax_path.axhline(0.5, color=C_RAND, ls="--", lw=1.2, alpha=0.8, label="Random (0.50)")
    ax_path.set_ylim(0, 1.05)
    ax_path.legend(fontsize=9)
    ax_path.grid(True, alpha=0.3)

    # ── 2. Total delay per HO ─────────────────────────────────────────────────
    ax_del.set_title("End-to-End Delay per Handover")
    ax_del.set_xlabel("Handover index")
    ax_del.set_ylabel("Total delay (ms)")

    if has_rand:
        ax_del.scatter(rand_ho_ids, rand_total, c=C_RAND, s=10, alpha=0.35,
                       zorder=2, label="_nolegend_")
        if nr >= 10:
            win_r = min(20, nr // 4)
            ra_r  = np.convolve(rand_total, np.ones(win_r) / win_r, mode="valid")
            ax_del.plot(rand_ho_ids[win_r - 1:], ra_r, color=C_RAND2, lw=1.5,
                        ls="--", label=f"Random avg ({win_r})", zorder=3)

    ax_del.scatter(ho_ids, total_ms, c=colours, s=14, alpha=0.7, zorder=4)
    ax_del.plot(ho_ids, oracle, color=C_BEST, lw=1.2, ls="--", label="Oracle best", zorder=5)
    if n >= 10:
        win = min(20, n // 4)
        ra  = np.convolve(total_ms, np.ones(win) / win, mode="valid")
        ax_del.plot(ho_ids[win - 1:], ra, color="black", lw=1.5,
                    label=f"Bregman avg ({win})", zorder=5)

    legend_handles = ax_del.get_legend_handles_labels()[0] + [
        mpatches.Patch(color=C_ISL, label="ISL (Bregman)"),
        mpatches.Patch(color=C_GND, label="GND (Bregman)"),
    ]
    if has_rand:
        legend_handles.append(mpatches.Patch(color=C_RAND, label="Random"))
    ax_del.legend(fontsize=8, handles=legend_handles)
    ax_del.grid(True, alpha=0.3)

    # ── 3. Mean latency breakdown per path ────────────────────────────────────
    ax_bar.set_title("Mean Latency Breakdown by Path")
    ax_bar.set_ylabel("Delay (ms)")

    components = [
        ("Propagation", prop_ms,  "#90CAF9"),
        ("AMF",         amf_ms,   "#1565C0"),
        ("SMF",         smf_ms,   "#FF8A65"),
        ("UPF",         upf_ms,   "#BF360C"),
    ]

    if has_rand:
        # Four groups: ISL-Bregman, ISL-Random, GND-Bregman, GND-Random
        group_labels = ["ISL\nBregman", "ISL\nRandom", "GND\nBregman", "GND\nRandom"]
        masks_algo   = [isl_mask,      rand_isl_mask,  gnd_mask,       rand_gnd_mask]
        arrs_prop    = [prop_ms, rand_prop, prop_ms, rand_prop]
        arrs_amf     = [amf_ms,  rand_amf,  amf_ms,  rand_amf]
        arrs_smf     = [smf_ms,  rand_smf,  smf_ms,  rand_smf]
        arrs_upf     = [upf_ms,  rand_upf,  upf_ms,  rand_upf]
        comp_arrs    = [arrs_prop, arrs_amf, arrs_smf, arrs_upf]
    else:
        group_labels = ["ISL", "Ground"]
        masks_algo   = [isl_mask, gnd_mask]
        comp_arrs    = [[prop_ms, prop_ms], [amf_ms, amf_ms],
                        [smf_ms, smf_ms],  [upf_ms, upf_ms]]

    x       = np.arange(len(group_labels))
    w       = 0.55
    bottoms = np.zeros(len(group_labels))

    for (label, _, colour), arrs in zip(components, comp_arrs):
        means = np.array([
            arrs[gi][masks_algo[gi]].mean() if masks_algo[gi].any() else 0.0
            for gi in range(len(group_labels))
        ])
        bars = ax_bar.bar(x, means, w, bottom=bottoms, color=colour, label=label,
                          edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, means):
            if val > 1:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + val / 2,
                            f"{val:.1f}", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
        bottoms += means

    for xi, (gl, mask) in enumerate(zip(group_labels, masks_algo)):
        cnt = mask.sum()
        ax_bar.text(xi, bottoms[xi] + 1, f"n={cnt}", ha="center", va="bottom", fontsize=7.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(group_labels, fontsize=8)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)

    # ── 4. Ground core — per-layer instance probabilities ────────────────────
    ax_gnd.set_title("Ground Core — Instance Probabilities")
    ax_gnd.set_xlabel("Handover index")
    ax_gnd.set_ylabel("Selection probability")

    styles = ["-", "--", ":"]
    for i, (p, style) in enumerate(zip(gnd_amf_p, styles)):
        ax_gnd.plot(ho_ids, p, color="#1565C0", ls=style, lw=1.2, label=f"AMF-{i}")
    for i, (p, style) in enumerate(zip(gnd_smf_p, styles)):
        ax_gnd.plot(ho_ids, p, color="#FF5722", ls=style, lw=1.2, label=f"SMF-{i}")
    for i, (p, style) in enumerate(zip(gnd_upf_p, styles)):
        ax_gnd.plot(ho_ids, p, color="#4CAF50", ls=style, lw=1.2, label=f"UPF-{i}")

    ax_gnd.axhline(1 / 3, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax_gnd.set_ylim(0, 1.05)
    ax_gnd.legend(fontsize=7, ncol=3)
    ax_gnd.grid(True, alpha=0.3)

    # ── 5. Onboard core — per-layer instance probabilities ───────────────────
    ax_on.set_title("Onboard Core — Instance Probabilities")
    ax_on.set_xlabel("Handover index")
    ax_on.set_ylabel("Selection probability")

    for i, (p, style) in enumerate(zip(on_amf_p, styles)):
        ax_on.plot(ho_ids, p, color="#1565C0", ls=style, lw=1.2, label=f"AMF-{i}")
    for i, (p, style) in enumerate(zip(on_smf_p, styles)):
        ax_on.plot(ho_ids, p, color="#FF5722", ls=style, lw=1.2, label=f"SMF-{i}")
    for i, (p, style) in enumerate(zip(on_upf_p, styles)):
        ax_on.plot(ho_ids, p, color="#4CAF50", ls=style, lw=1.2, label=f"UPF-{i}")

    ax_on.axhline(1 / 3, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax_on.set_ylim(0, 1.05)
    ax_on.legend(fontsize=7, ncol=3)
    ax_on.grid(True, alpha=0.3)

    # ── 6. Cumulative regret ──────────────────────────────────────────────────
    ax_reg.set_title("Cumulative Regret vs Handover Index")
    ax_reg.set_xlabel("Handover index")
    ax_reg.set_ylabel("Cumulative regret (ms)")
    ax_reg.plot(ho_ids, cum_reg, color="black", lw=1.5, label="Bregman")

    if has_rand:
        ax_reg.plot(rand_ho_ids, rand_cum_reg, color=C_RAND2, lw=1.5,
                    ls="--", label="Random baseline")

    # Linear reference at Bregman's own rate (visual guide only)
    if n > 1:
        linear_ref = cum_reg[-1] / n * np.array(ho_ids)
        ax_reg.plot(ho_ids, linear_ref, color="red", ls=":", lw=1.0,
                    alpha=0.5, label="Linear ref (Bregman rate)")

    ax_reg.legend(fontsize=9)
    ax_reg.grid(True, alpha=0.3)

    # ── Summary ───────────────────────────────────────────────────────────────
    isl_count = isl_mask.sum()
    gnd_count = gnd_mask.sum()
    print(f"\n{'═'*60}")
    print(f"  Dispatcher results summary  ({n} HOs)")
    print(f"{'═'*60}")
    print(f"  Bregman ISL selected : {isl_count:>4}  ({100*isl_count/n:.1f}%)")
    print(f"  Bregman GND selected : {gnd_count:>4}  ({100*gnd_count/n:.1f}%)")
    print(f"{'─'*60}")
    for label, mask in [("ISL (Bregman)", isl_mask), ("GND (Bregman)", gnd_mask)]:
        if not mask.any():
            continue
        t = total_ms[mask]
        print(f"  {label} : avg={t.mean():.2f} ms  "
              f"min={t.min():.2f} ms  max={t.max():.2f} ms")
    if has_rand:
        print(f"{'─'*60}")
        ri = rand_isl_mask.sum()
        rg = rand_gnd_mask.sum()
        print(f"  Random  ISL selected : {ri:>4}  ({100*ri/nr:.1f}%)")
        print(f"  Random  GND selected : {rg:>4}  ({100*rg/nr:.1f}%)")
        for label, mask, arrs in [
            ("ISL (Random)", rand_isl_mask, rand_total),
            ("GND (Random)", rand_gnd_mask, rand_total),
        ]:
            if not mask.any():
                continue
            t = arrs[mask]
            print(f"  {label} : avg={t.mean():.2f} ms  "
                  f"min={t.min():.2f} ms  max={t.max():.2f} ms")
    print(f"{'─'*60}")
    print(f"  Oracle avg delay       : {oracle.mean():.2f} ms")
    print(f"  Bregman avg per-HO reg : {(cum_reg[-1]/n):.2f} ms  "
          f"(final={cum_reg[-1]:.1f} ms)")
    if has_rand:
        print(f"  Random  avg per-HO reg : {(rand_cum_reg[-1]/nr):.2f} ms  "
              f"(final={rand_cum_reg[-1]:.1f} ms)")
    print(f"{'═'*60}\n")

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {OUT_PNG}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
