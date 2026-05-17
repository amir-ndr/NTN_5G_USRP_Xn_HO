# USRP Testing Guide — 6-Experiment Automation

## Quick Start

```bash
# Run all 6 experiments sequentially
bash run_experiments.sh
```

## What Gets Run

The script automates these 6 experiments:

| # | Tau Profile | Load Scale | Description |
|---|-------------|-----------|-------------|
| 1 | default | 0.5 | Light load |
| 2 | default | 1.0 | Default load |
| 3 | default | 1.5 | Heavy load |
| 4 | default | 2.0 | Extreme load |
| 5 | strict | 1.0 | Strict τ_max profile |
| 6 | relaxed | 1.0 | Relaxed τ_max profile |

**Each run:** 300 HOs (HO_HARD_CAP = 300)

---

## How to Use With USRPs

### 1. **Start USRP Receivers/Transmitters**

Before running the script, have your 3 USRPs ready:

```bash
# In USRP terminal 1 (trgSAT transmitter)
python trgSAT.py

# In USRP terminal 2 (TN receiver)
python TN.py

# In USRP terminal 3 (your choice - can be another receiver or delay injector)
```

### 2. **Run the Experiment Script**

In a separate terminal (or tmux session):

```bash
cd /Users/amir/Desktop/Queens/Network/DND/NTN_5G_USRP_Xn_HO
bash run_experiments.sh
```

The script will:
- Run experiment 1 (default_ls0.5) until 300 HOs complete
- Automatically pause 5 seconds
- Move to experiment 2 (default_ls1.0)
- Continue until all 6 are done
- Print timestamps and progress for each

### 3. **Monitor Progress**

Each run logs to: `results/<tau>_ls<scale>/controller.log`

View in real-time with:
```bash
# In another terminal, watch the latest log
tail -f results/default_ls0p5/controller.log
```

---

## Resuming If Interrupted

If a run crashes or you need to pause:

```bash
# Resume from experiment 3 (strict_ls1.0)
bash run_experiments.sh --resume 3
```

The script tracks state in `.experiment_state` file.

---

## After All Runs Complete

Generate the comprehensive analysis plots:

```bash
python plot_all.py --selections
```

This creates:
- `results_all/exp1/` — Regret analysis
- `results_all/exp2/` — Instance probabilities
- `results_all/exp3/` — Task sensitivity
- `results_all/exp4/` — Load sweep
- `results_all/exp5/` — Tau sensitivity
- `results_all/exp6/` — Path selection
- `results_all/exp7/` — Baseline comparisons
- `results_all/selections/` — Selection timelines per experiment

---

## Expected Output Structure

```
results/
├── default_ls0p5/
│   ├── dispatch_log_default_ls0p5.csv    ← Main results
│   ├── random_log_default_ls0p5.csv
│   ├── greedy_log_default_ls0p5.csv
│   ├── controller.log                    ← Run transcript
│   └── [other output files]
├── default_ls1/
├── default_ls1p5/
├── default_ls2/
├── strict_ls1/
└── relaxed_ls1/

results_all/
├── exp1/ — Regret convergence
├── exp2/ — Probability trajectories
├── ... (7 experiments total)
└── selections/ — Path & instance selection timelines
```

---

## Monitoring Tips

### Real-time HO count
```bash
# Check progress of current experiment
wc -l results/*/dispatch_log_*.csv | sort -n | tail -1
```

### Watch for errors
```bash
# Grep for errors in all logs
grep -i "error\|fail\|exception" results/*/controller.log
```

### Resource usage
```bash
# Monitor Python process
top -p $(pgrep -f "python controller.py")
```

---

## Troubleshooting

**Q: Script stops after first run**
- A: Check if controller.py exited with error
- Look at: `results/default_ls0p5/controller.log`

**Q: HO count is lower than expected**
- A: USRP may have disconnected or timed out
- Check USRP terminal for transmission/reception errors
- You can resume from that experiment: `bash run_experiments.sh --resume N`

**Q: How long will all 6 take?**
- A: ~6-10 hours depending on orbital geometry and handover frequency
- Each run with 300 HOs typically takes 1-2 hours

---

## Parameters Used

All runs use these optimized parameters (already set in `dispatcher.py`):

```python
ETA_PATH     = 0.5      # Fast path learning
PROB_FLOOR   = 0.03     # Wide path distribution
TARGET_RHO   = 0.45     # Strong exploration gradient
ETA_B_PATH   = 0.001    # Stable exploitation
grad_B_cap   = 0.5      # Adaptive B learning
```

---

## Questions?

- Check `controller.py` for detailed algorithm setup
- See `dispatcher.py` for parameter definitions
- Review dispatch logs (CSV) for per-HO data

**Good luck with USRP testing!** 🚀
