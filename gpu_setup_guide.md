# Setting Up GPU Environment for Hybrid GeoAI Building Footprint Project

This guide will walk you through setting up a GPU-accelerated environment for running the Hybrid GeoAI Building Footprint regularization project with PyTorch.

## Prerequisites

- NVIDIA GPU with CUDA support
- Windows 10/11 or compatible OS
- Anaconda or Miniconda (recommended for environment management)
- Git (for version control)

## Step 1: Install NVIDIA Drivers and CUDA Toolkit

1. Download and install the latest NVIDIA drivers for your GPU from the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page.

2. Download and install the CUDA Toolkit 11.8 (recommended for compatibility with PyTorch) from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

3. Verify your CUDA installation by running:
   ```bash
   nvcc --version
   ```

## Step 2: Create a GPU-Enabled Conda Environment

```bash
# Create a new environment
conda create -n geoai-gpu python=3.10
conda activate geoai-gpu

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Step 3: Verify GPU Detection

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Step 4: Install Required Dependencies

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd <repository-directory>

# Install required packages
pip install -r requirements-gpu.txt
```

## Step 5: Project-Specific GPU Configuration

Create a `gpu_config.py` file in your project directory with the following content:

```python
import torch

# GPU configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = torch.cuda.device_count() > 1
GPU_BATCH_SIZE = 16  # Adjust based on your GPU memory
GPU_WORKERS = 4      # Parallel data loading

# Training parameters optimized for GPU
GPU_LEARNING_RATE = 2e-4
GPU_WEIGHT_DECAY = 1e-4

# Precision settings
USE_MIXED_PRECISION = True
PRECISION_DTYPE = torch.float16 if USE_MIXED_PRECISION else torch.float32

# Performance optimizations
CUDNN_BENCHMARK = True
torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
```

## Step 6: Running with GPU Acceleration

To run the project with GPU acceleration:

```bash
# Activate the GPU environment
conda activate geoai-gpu

# Run with GPU flag
python main.py --use-gpu --multi-state --n-states 10
```

## Troubleshooting

### CUDA Out of Memory Errors
If you encounter CUDA out of memory errors:
- Reduce batch size in `gpu_config.py`
- Enable gradient checkpointing for large models
- Use mixed precision training

### Slow GPU Performance
- Ensure your GPU drivers are up to date
- Check for thermal throttling with `nvidia-smi` monitoring
- Verify your model is actually using the GPU by monitoring usage

### Version Compatibility Issues
- Make sure your CUDA version is compatible with your PyTorch version
- Check compatibility matrix at [PyTorch installation page](https://pytorch.org/get-started/locally/)