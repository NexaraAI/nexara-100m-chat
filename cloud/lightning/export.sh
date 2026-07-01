#!/bin/bash
set -e

# Export checkpoint for inference
PYTHON_PATH="/home/zeus/miniconda3/envs/cloudspace/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python"
fi

echo "=== Exporting Nexara checkpoint ==="
$PYTHON_PATH scripts/export_checkpoint.py "$@"
