#!/usr/bin/env python3
"""
plot_tau_profiles.py — τ_max profile comparison plots

Reads CSVs from results/{profile}_ls1/ (load scale = 1.0, the default).
Each profile run must have been produced by:
    python3 controller.py --tau-profile default  --load-scale 1.0
    python3 controller.py --tau-profile strict   --load-scale 1.0
    python3 controller.py --tau-profile relaxed  --load-scale 1.0

Generates:
  1. Cumulative global regret vs HO index (all 3 profiles + random baseline)
  2. Running-average end-to-end latency vs HO index (all 3 profiles + random)
  3. Path probability (π_ISL) vs HO index (all 3 profiles)
  4. Ground core instance selection probability — one figure per τ profile
  5. Onboard core instance selection probability — one figure per τ profile
  6. UPF instance selection — 2×3 panel (Ground/Onboard × 3 profiles)

Output: tau_result/
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUT_DIR     = HERE / "tau_result"
OUT_DIR.mkdir(exist_ok=True)

# Load scale used for the τ_max comparison (always 1.0 → suffix "ls1").
# Adjust LS_TAG here if you ran the profiles at a different --load-scale.
LS_TAG = "ls1"

def _csv(profile: str, prefix: str) -> Path:
    tag = f"{profile}_{LS_TAG}"
    return RESULTS_DIR / tag / f"{prefix}_{tag}.csv"

FILES = {
    "default": _csv("default", "dispatch_log"),
    "strict":  _csv("strict",  "dispatch_log"),
    "relaxed": _csv("relaxed", "dispatch_log"),
}
RANDOM_FILE = _csv("default", "random_log")

# Verify all required files exist before attempting to load anything.
missing = [(name, path) for name, path in {**FILES, "random": RANDOM_FILE}.items()
           if not path.exists()]
if missing:
    print("ERROR: the following CSV files were not found:", file=sys.stderr)
    for name, path in missing:
        print(f"  [{name}]  {path}", file=sys.stderr)
    print("\nRun the three profile simulations first:", file=sys.stderr)
    for profile in ["default", "strict", "relaxed"]:
        print(f"  python3 controller.py --tau-profile {profile} --load-scale 1.0",
              file=sys.stderr)
    sys.exit(1)

# ── style ──────────────────────────────────────────────────────────────────────
PROFILE_COLOR = {"default": "#2196F3", "strict": "#F44336", "relaxed": "#4CAF50"}
PROFILE_LABEL = {"default": "Default τ", "strict": "Strict τ",  "relaxed": "Relaxed τ"}
RANDOM_COLOR  = "#9E9E9E"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 1.8,
})

# ── load data ──────────────────────────────────────────────────────────────────
dfs = {name: pd.read_csv(path) for name, path in FILES.items()}
df_rand = pd.read_csv(RANDOM_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Cumulative global regret
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for name, df in dfs.items():
    ax.plot(df["ho_id"], df["cum_global_regret_ms"],
            color=PROFILE_COLOR[name], label=PROFILE_LABEL[name])

# random baseline uses its own cum column if present, else cumsum
if "cum_global_regret_ms" in df_rand.columns:
    ax.plot(df_rand["ho_id"], df_rand["cum_global_regret_ms"],
            color=RANDOM_COLOR, linestyle="--", label="Random baseline")
else:
    ax.plot(df_rand["ho_id"], df_rand["global_regret_ms"].cumsum(),
            color=RANDOM_COLOR, linestyle="--", label="Random baseline")

ax.set_xlabel("Handover index")
ax.set_ylabel("Cumulative global regret (ms)")
ax.set_title("Cumulative Global Regret vs Handover Index")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "1_cumulative_regret.png")
plt.close(fig)
print("Saved: 1_cumulative_regret.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Running-average end-to-end latency
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for name, df in dfs.items():
    running_avg = df["total_ms"].expanding().mean()
    ax.plot(df["ho_id"], running_avg,
            color=PROFILE_COLOR[name], label=PROFILE_LABEL[name])

rand_avg = df_rand["total_ms"].expanding().mean()
ax.plot(df_rand["ho_id"], rand_avg,
        color=RANDOM_COLOR, linestyle="--", label="Random baseline")

ax.set_xlabel("Handover index")
ax.set_ylabel("Running-average latency (ms)")
ax.set_title("Running-Average End-to-End Latency vs Handover Index")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "2_running_avg_latency.png")
plt.close(fig)
print("Saved: 2_running_avg_latency.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Path probability (π_ISL) vs HO index — all three profiles
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

for name, df in dfs.items():
    ax.plot(df["ho_id"], df["path_p_isl"],
            color=PROFILE_COLOR[name], label=PROFILE_LABEL[name], alpha=0.85)

ax.axhline(0.5, color=RANDOM_COLOR, linestyle="--", linewidth=1.2, label="Random (0.5)")
ax.set_xlabel("Handover index")
ax.set_ylabel("π_ISL  (probability of choosing ISL path)")
ax.set_title("ISL Path Selection Probability vs Handover Index")
ax.set_ylim(-0.02, 1.02)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "3_path_probability.png")
plt.close(fig)
print("Saved: 3_path_probability.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4 & 5. Per-profile NF instance selection probabilities (AMF/SMF/UPF, 1×3 each)
# ══════════════════════════════════════════════════════════════════════════════

NF_LAYERS = ["amf", "smf", "upf"]
NF_LABEL  = {"amf": "AMF", "smf": "SMF", "upf": "UPF"}
INST_COLORS = ["#1565C0", "#EF6C00", "#2E7D32"]

for name, df in dfs.items():
    for side, side_tag, side_title in [
        ("gnd", "4", "Ground"),
        ("on",  "5", "Onboard (NTN)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
        fig.suptitle(
            f"{side_title} Core Instance Selection Probability — {PROFILE_LABEL[name]}",
            fontsize=13, y=1.02,
        )

        for ax, nf in zip(axes, NF_LAYERS):
            col0 = f"{side}_{nf}_p0"
            if col0 not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(NF_LABEL[nf])
                continue

            ho = df["ho_id"]
            for k in range(3):
                ax.plot(ho, df[f"{side}_{nf}_p{k}"],
                        color=INST_COLORS[k], label=f"Instance {k}", alpha=0.88)

            ax.set_title(NF_LABEL[nf])
            ax.set_xlabel("Handover index")
            ax.set_ylabel("Selection probability" if nf == "amf" else "")
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"{side_tag}_{side}_instance_probs_{name}.png"
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. UPF instance selection — 2×3 panel (Ground/Onboard × Default/Strict/Relaxed)
# ══════════════════════════════════════════════════════════════════════════════

PROFILES     = ["default", "strict", "relaxed"]
SIDE_INFO    = [("gnd", "Ground UPF"), ("on", "Onboard UPF")]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
fig.suptitle("UPF Instance Selection Probability", fontsize=14, y=1.01)

for row, (side, row_label) in enumerate(SIDE_INFO):
    for col, profile in enumerate(PROFILES):
        ax = axes[row, col]
        df = dfs[profile]
        ho = df["ho_id"]

        for k in range(3):
            ax.plot(ho, df[f"{side}_upf_p{k}"],
                    color=INST_COLORS[k], label=f"Instance {k}",
                    linewidth=1.6, alpha=0.88)

        if row == 0:
            ax.set_title(PROFILE_LABEL[profile], fontsize=12, pad=6)
        if col == 0:
            ax.set_ylabel(f"{row_label}\nSelection probability", fontsize=10)
        ax.set_xlabel("Handover index", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 2:
            ax.legend(fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "6_upf_instance_probs.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved: 6_upf_instance_probs.png")

print(f"\nAll figures written to: {OUT_DIR}/")
