# Start from NVIDIA's PyTorch container which already has CUDA and PyTorch configured optimally
# Specify a specific version for better reproducibility
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    HF_DATASETS_CACHE=/tmp/datasets_cache \
    PYTHON_VERSION=3.10

# Install basic dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file - note this is from the docker context, not a relative path
COPY requirements.txt /workspace/requirements.txt

# Install dependencies from requirements file
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Create working directories
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs /workspace/outputs

# Set up environment for better GPU utilization
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_DEBUG=INFO \
    NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=0 \
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24

# Add A100-specific optimizations
ENV TORCH_CUDNN_V8_API_ENABLED=1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_BUFFSIZE=16777216 \
    CUDA_AUTO_BOOST=0

# Install additional packages for monitoring
RUN pip install --no-cache-dir \
    py-spy \
    psutil \
    gpustat

# Copy entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Verify Python version
RUN python --version && \
    pip --version

# Set the entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]