#!/bin/bash
set -e

# Resume training from latest checkpoint
PYTHON_PATH="/home/zeus/miniconda3/envs/cloudspace/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python"
fi

echo "=== Resuming Nexara stage 1 training ==="
$PYTHON_PATH scripts/train_long.py --config configs/stage1_tinystories.toml "$@"
