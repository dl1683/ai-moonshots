#!/bin/bash
# Switch from v0.5.2 to v0.5.3 production training
# Usage: bash code/switch_to_v053.sh

REPO="C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra"
cd "$REPO"

echo "=== Switching to v0.5.3 ==="

# 1. Find latest v0.5.2 checkpoint
CKPT=$(ls -t results/checkpoints_v052/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No v0.5.2 checkpoint found!"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# 2. Create warm-start
python -c "
import sys; sys.path.insert(0, 'code')
from launch_v053 import warmstart_v053
import torch
model = warmstart_v053('$CKPT')
torch.save(model.state_dict(), 'results/v053_warmstart.pt')
print('Saved v0.5.3 warm-start')
"

# 3. Update trainer to use v0.5.3
sed -i 's/from launch_v052 import create_v052 as _create_model/from launch_v053 import create_v053 as _create_model/' code/sutra_v05_train.py
sed -i 's/checkpoints_v052/checkpoints_v053/g' code/sutra_v05_train.py
sed -i 's/v052_log/v053_log/g' code/sutra_v05_train.py
sed -i 's/v052_metrics/v053_metrics/g' code/sutra_v05_train.py
sed -i 's/v052_best/v053_best/g' code/sutra_v05_train.py
sed -i 's/v052_warmstart/v053_warmstart/g' code/sutra_v05_train.py

echo "=== Ready. Launch with: python -u code/sutra_v05_train.py 2>&1 | tee results/v053_log.txt ==="
