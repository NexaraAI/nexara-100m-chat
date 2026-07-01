#!/bin/bash
set -e

# Run training
PYTHON_PATH="/home/zeus/miniconda3/envs/cloudspace/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python"
fi

echo "=== Running Nexara stage 1 training ==="
$PYTHON_PATH scripts/train_long.py --config configs/stage1_tinystories.toml "$@"
