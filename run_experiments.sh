#!/bin/bash
###############################################################################
# run_experiments.sh — Automate 6-experiment sequence for USRP testing
#
# Usage:
#   bash run_experiments.sh              # Run all 6 in sequence
#   bash run_experiments.sh --resume 3   # Resume from experiment 3
#
# Each experiment runs until HO_HARD_CAP=300 HOs are reached.
# Output logged to results/<tau>_ls<scale>/controller.log
###############################################################################

set -e  # Exit on error

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Experiment definitions (tau_profile, load_scale)
EXPERIMENTS=(
    "default:0.5"
    "default:1.0"
    "default:1.5"
    "default:2.0"
    "strict:1.0"
    "relaxed:1.0"
)

# State file for resuming
STATE_FILE=".experiment_state"

# Function to print banner
banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# Function to print status
status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

# Function to print success
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Determine starting experiment
START_EXP=0
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    START_EXP=$((${2} - 1))
    if [ -f "$STATE_FILE" ]; then
        rm "$STATE_FILE"
    fi
fi

# Main loop
TOTAL=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
    if [ $i -lt $START_EXP ]; then
        continue
    fi

    EXP="${EXPERIMENTS[$i]}"
    TAU="${EXP%:*}"
    LOAD="${EXP#*:}"
    EXP_NUM=$((i + 1))

    banner "Experiment $EXP_NUM/$TOTAL: tau=$TAU  load=$LOAD"

    status "Starting controller..."
    echo "tau=$TAU, load=$LOAD, exp=$EXP_NUM" > "$STATE_FILE"

    # Create results directory if needed
    RESULTS_DIR="results/${TAU}_ls$(echo $LOAD | tr '.' 'p')"
    mkdir -p "$RESULTS_DIR"
    LOG_FILE="$RESULTS_DIR/controller.log"

    # Run controller and capture output
    start_time=$(date +%s)
    if python controller.py \
        --tau-profile "$TAU" \
        --load-scale "$LOAD" \
        2>&1 | tee "$LOG_FILE"; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))

        # Count HOs in output
        ho_count=$(wc -l < "$RESULTS_DIR/dispatch_log_${TAU}_ls$(echo $LOAD | tr '.' 'p').csv" 2>/dev/null || echo "0")
        ho_count=$((ho_count - 1))  # Subtract header

        success "Experiment $EXP_NUM completed!"
        status "  HOs captured: $ho_count"
        status "  Duration: ${duration}s"
        status "  Log: $LOG_FILE"
    else
        error "Experiment $EXP_NUM FAILED!"
        error "  Check log: $LOG_FILE"
        exit 1
    fi

    # Summary before next experiment
    if [ $i -lt $((TOTAL - 1)) ]; then
        NEXT_EXP="${EXPERIMENTS[$((i + 1))]}"
        NEXT_TAU="${NEXT_EXP%:*}"
        NEXT_LOAD="${NEXT_EXP#*:}"

        echo ""
        warning "Ready for next experiment in 5 seconds..."
        status "Next: tau=$NEXT_TAU  load=$NEXT_LOAD"

        # Wait with countdown
        for s in {5..1}; do
            echo -ne "  ${s}...\r"
            sleep 1
        done
        echo "          "  # Clear countdown line
    fi
done

# All done
rm -f "$STATE_FILE"
banner "All 6 experiments completed! ✓"
status "Results are in: results/"
status "Generate plots with: python plot_all.py --selections"
