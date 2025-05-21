# phi-4-finetuning/docker/entrypoint.sh

#!/bin/bash
set -e

# Configure environment based on GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $GPU_COUNT GPUs"

# Optimize settings based on GPU count
if [ $GPU_COUNT -gt 1 ]; then
    # Multi-GPU setup
    export NCCL_P2P_LEVEL=NVL
    echo "Configuring for multi-GPU training with $GPU_COUNT GPUs"
else
    # Single GPU setup
    echo "Configuring for single GPU training"
fi

# Configure memory optimization for A100 GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print system info
echo "=== System Information ==="
nvidia-smi
echo "=========================="

# Run the provided command or default to bash
if [ $# -eq 0 ]; then
    echo "No command provided, starting bash..."
    exec /bin/bash
else
    echo "Running command: $@"
    exec "$@"
fi