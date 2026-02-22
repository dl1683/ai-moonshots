#!/bin/bash
# Run this script AFTER surgery completes to analyze and launch next experiments
# Usage: bash run_after_surgery.sh

cd "/c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-fractal-embeddings"

echo "=== Post-Surgery Analysis Pipeline ==="
echo "Step 1: Running surgery post-hoc analysis..."
python src/cti_surgery_analysis.py
echo "Surgery analysis done."

echo ""
echo "Step 2: Launching deff_formula_validation..."
# Clear old logs
rm -f results/cti_deff_formula_log.txt results/cti_deff_formula_validation.json
python -u src/cti_deff_formula_validation.py &
DEFF_FORMULA_PID=$!
echo "deff_formula_validation started (bg)"

echo ""
echo "Step 3: Launching deff_signal_validation..."
rm -f results/cti_deff_signal_log.txt results/cti_deff_signal_validation.json  
python -u src/cti_deff_signal_validation.py &
DEFF_SIGNAL_PID=$!
echo "deff_signal_validation started (bg)"

echo ""
echo "PIDs: deff_formula=$DEFF_FORMULA_PID, deff_signal=$DEFF_SIGNAL_PID"
echo "Monitor with: tail -f results/cti_deff_formula_log.txt"
