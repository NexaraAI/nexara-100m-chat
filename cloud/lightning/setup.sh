#!/bin/bash
set -e

echo "=== Setting up Nexara Remote Studio environment ==="

# 1. Determine python path
PYTHON_PATH="/home/zeus/miniconda3/envs/cloudspace/bin/python"
PIP_PATH="/home/zeus/miniconda3/envs/cloudspace/bin/pip"

if [ ! -f "$PYTHON_PATH" ]; then
    echo "Conda environment 'cloudspace' not found at $PYTHON_PATH"
    echo "Falling back to system python..."
    PYTHON_PATH="python"
    PIP_PATH="pip"
fi

# 2. Install nexara package in editable mode
echo "Installing nexara package in editable mode..."
$PIP_PATH install -e .

# 3. Verify PyTorch CUDA installation
echo "Verifying CUDA availability..."
$PYTHON_PATH -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Model:', torch.cuda.get_device_name(0))
    print('CUDA Version:', torch.version.cuda)
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory / (1024**3), 'GB')
"

echo "=== Setup completed successfully ==="
