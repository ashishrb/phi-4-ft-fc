# Start from NVIDIA's PyTorch container which already has CUDA and PyTorch configured optimally
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

# Install packages with versions compatible with the base image's CUDA stack
RUN pip install --no-cache-dir \
    transformers==4.48.0 \
    datasets==3.3.0 \
    tokenizers==0.21.0 \
    accelerate==0.27.2 \
    evaluate==0.4.1 \
    peft==0.14.0 \
    bitsandbytes==0.41.3 \
    deepspeed==0.13.1 \
    tensorboard==2.15.1 \
    scikit-learn==1.4.2 \
    "numpy>=1.21,<1.25" \
    "pandas>=1.5.3,<1.6.0" \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    tqdm==4.66.3 \
    rich==13.7.0 \
    coloredlogs==15.0.1 \
    matplotlib \
    plotly \
    Jinja2 \
    einops==0.8.0 \
    py-spy==0.3.14 \
    gpustat==1.1.1

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

# Copy entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Verify installation and Python version
RUN python --version && \
    pip --version && \
    python -c "import torch, transformers, datasets, accelerate; print('✅ Core imports successful'); print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Set the entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]