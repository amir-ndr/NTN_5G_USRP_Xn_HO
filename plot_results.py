#!/usr/bin/env python3
"""
plot_results.py — Post-session results plotter

Reads trgSAT_log.csv and TN_log.csv and produces a 4-panel figure:
  1. Propagation delay over time (both nodes, timeline)
  2. Packets received per node (bar chart — selection count after HO)
  3. Latency distribution histogram (both nodes overlaid)
  4. Cumulative average latency over time (convergence)

Usage:
    python3 plot_results.py
    python3 plot_results.py --no-show      # save PNG only, don't open window
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

_HERE = Path(__file__).resolve().parent

TRG_CSV = _HERE / "trgSAT_log.csv"
TN_CSV  = _HERE / "TN_log.csv"
OUT_PNG = _HERE / "results.png"

# colours
C_TRG = "#2196F3"   # blue  — trgSAT
C_TN  = "#FF5722"   # orange — TN


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[plot] WARNING: {path} not found — skipping.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_records(rows: list[dict]) -> tuple[list[datetime], list[float]]:
    times, delays = [], []
    for r in rows:
        try:
            t = datetime.fromisoformat(r["timestamp"])
            d = float(r["prop_delay_ms"])
            times.append(t)
            delays.append(d)
        except (KeyError, ValueError):
            continue
    return times, delays


def main():
    p = argparse.ArgumentParser(description="Plot trgSAT and TN session results")
    p.add_argument("--no-show", action="store_true", help="Save PNG only, do not open window")
    args = p.parse_args()

    trg_rows = load_csv(TRG_CSV)
    tn_rows  = load_csv(TN_CSV)

    if not trg_rows and not tn_rows:
        print("[plot] No data found. Run trgSAT.py and TN.py first, then re-run this script.")
        sys.exit(1)

    trg_times, trg_delays = parse_records(trg_rows)
    tn_times,  tn_delays  = parse_records(tn_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("USRP BPSK — Session Results", fontsize=14, fontweight="bold")
    ax_time, ax_bar, ax_hist, ax_cum = axes.flat

    # ── 1. Propagation delay over time ───────────────────────────────────────
    ax_time.set_title("Propagation Delay Over Time")
    ax_time.set_xlabel("Time (UTC)")
    ax_time.set_ylabel("Delay (ms)")

    if trg_times:
        ax_time.plot(trg_times, trg_delays, "o-", color=C_TRG, ms=4, lw=1.2,
                     label=f"trgSAT (n={len(trg_delays)})", alpha=0.85)
    if tn_times:
        ax_time.plot(tn_times, tn_delays, "s-", color=C_TN, ms=4, lw=1.2,
                     label=f"TN (n={len(tn_delays)})", alpha=0.85)

    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)
    ax_time.legend(fontsize=9)
    ax_time.grid(True, alpha=0.3)

    # ── 2. Packets received per node (HO selection count) ────────────────────
    ax_bar.set_title("Packets Received per Node\n(HO selection count)")
    ax_bar.set_ylabel("Packet count")

    nodes   = ["trgSAT", "TN"]
    counts  = [len(trg_delays), len(tn_delays)]
    colours = [C_TRG, C_TN]
    bars    = ax_bar.bar(nodes, counts, color=colours, width=0.45, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=11)

    total = sum(counts)
    if total > 0:
        for bar, count, node in zip(bars, counts, nodes):
            pct = 100 * count / total
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        f"{pct:.1f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")

    ax_bar.set_ylim(0, max(counts, default=1) * 1.25)
    ax_bar.grid(axis="y", alpha=0.3)

    # ── 3. Latency distribution histogram ────────────────────────────────────
    ax_hist.set_title("Latency Distribution")
    ax_hist.set_xlabel("Propagation delay (ms)")
    ax_hist.set_ylabel("Count")

    all_delays = trg_delays + tn_delays
    if all_delays:
        bins = np.linspace(min(all_delays) - 1, max(all_delays) + 1, 30)
        if trg_delays:
            ax_hist.hist(trg_delays, bins=bins, color=C_TRG, alpha=0.65,
                         label=f"trgSAT  avg={sum(trg_delays)/len(trg_delays):.1f} ms")
        if tn_delays:
            ax_hist.hist(tn_delays, bins=bins, color=C_TN, alpha=0.65,
                         label=f"TN  avg={sum(tn_delays)/len(tn_delays):.1f} ms")

    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # ── 4. Cumulative average latency ─────────────────────────────────────────
    ax_cum.set_title("Cumulative Average Latency")
    ax_cum.set_xlabel("Packet index")
    ax_cum.set_ylabel("Running avg delay (ms)")

    if trg_delays:
        cum_trg = np.cumsum(trg_delays) / np.arange(1, len(trg_delays) + 1)
        ax_cum.plot(cum_trg, color=C_TRG, lw=1.5,
                    label=f"trgSAT  final={cum_trg[-1]:.2f} ms")
    if tn_delays:
        cum_tn = np.cumsum(tn_delays) / np.arange(1, len(tn_delays) + 1)
        ax_cum.plot(cum_tn, color=C_TN, lw=1.5,
                    label=f"TN  final={cum_tn[-1]:.2f} ms")

    ax_cum.legend(fontsize=9)
    ax_cum.grid(True, alpha=0.3)

    # ── Summary to terminal ───────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"  Results summary")
    print(f"{'═'*52}")
    for name, delays in [("trgSAT", trg_delays), ("TN", tn_delays)]:
        if delays:
            print(f"  {name:<8}  pkts={len(delays):>4}  "
                  f"avg={sum(delays)/len(delays):>7.2f} ms  "
                  f"min={min(delays):>7.2f} ms  "
                  f"max={max(delays):>7.2f} ms")
        else:
            print(f"  {name:<8}  pkts=   0  (no data)")
    total = len(trg_delays) + len(tn_delays)
    if total > 0:
        print(f"{'─'*52}")
        print(f"  Total packets : {total}")
        print(f"  trgSAT share  : {100*len(trg_delays)/total:.1f}%")
        print(f"  TN share      : {100*len(tn_delays)/total:.1f}%")
    print(f"{'═'*52}\n")

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {OUT_PNG}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
