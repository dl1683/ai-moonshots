#!/bin/bash
# POST-TRIPLET PIPELINE: launches experiments immediately when triplet arm completes
# Usage: bash src/cti_post_triplet_pipeline.sh [results_dir]
# Monitors for results/cti_cifar_triplet.json, then launches:
#   1. Anti-triplet arm (bidirectional causal test)
#   2. Cross-modal validation (frozen text law on vision)
#   After both: quantitative prediction

cd "$(dirname "$0")/.."
RESULTS_DIR="results"
TRIPLET_RESULT="$RESULTS_DIR/cti_cifar_triplet.json"

echo "=== POST-TRIPLET PIPELINE ==="
echo "Waiting for: $TRIPLET_RESULT"
echo "Will launch: anti-triplet arm + cross-modal validation"
echo ""

# Poll until triplet results exist
while [ ! -f "$TRIPLET_RESULT" ]; do
    echo "$(date): still waiting for triplet arm..."
    sleep 60
done

echo ""
echo "=== TRIPLET ARM COMPLETED ==="
echo "$(date): Found $TRIPLET_RESULT"
echo ""

# Show triplet summary
python -c "
import json
with open('$TRIPLET_RESULT') as f:
    d = json.load(f)
if 'summary' in d:
    s = d['summary']
    print(f'Triplet result: mean_q={s[\"mean_q\"]:.4f}  delta={s[\"delta_q\"]:+.4f}  passed={s[\"passed\"]}')
" 2>/dev/null

echo ""
echo "=== LAUNCHING ANTI-TRIPLET ARM ==="
python -u src/cti_cifar_antitriplet_arm.py > results/cti_antitriplet_run.log 2>&1 &
ANTI_PID=$!
echo "Anti-triplet arm PID: $ANTI_PID"

echo ""
echo "=== LAUNCHING CROSS-MODAL VALIDATION ==="
python -u src/cti_cifar_crossmodal_validation.py > results/cti_crossmodal_run.log 2>&1 &
CROSSMODAL_PID=$!
echo "Cross-modal validation PID: $CROSSMODAL_PID"

echo ""
echo "Waiting for anti-triplet arm and cross-modal validation to complete..."

# Wait for both
wait $ANTI_PID
echo "$(date): Anti-triplet arm done."
wait $CROSSMODAL_PID
echo "$(date): Cross-modal validation done."

echo ""
echo "=== RUNNING QUANTITATIVE PREDICTION ==="
python -u src/cti_quantitative_prediction.py 2>&1

echo ""
echo "=== ALL POST-TRIPLET EXPERIMENTS COMPLETE ==="
echo "Results:"
ls -la results/cti_cifar_triplet.json results/cti_cifar_antitriplet.json results/cti_cifar_crossmodal.json 2>/dev/null
