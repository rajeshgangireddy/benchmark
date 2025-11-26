# CUDA Environment Setup for Anomalib Benchmarking

This guide will help you set up a CUDA-enabled environment for running Anomalib benchmarks on NVIDIA GPUs.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed and configured
- Conda package manager installed

## Setup Instructions

### 1. Create a new conda environment

Create a fresh conda environment with Python 3.12.3:

```bash
conda create -n anomalib python==3.12.3
```

### 2. Activate the environment

```bash
conda activate anomalib
```

### 3. Verify NVIDIA drivers

Ensure your NVIDIA drivers are properly installed and the GPU is detected:

```bash
nvidia-smi
```

You should see your GPU information displayed. If this command fails, install or update your NVIDIA drivers first.

### 4. Install PyTorch with CUDA 13.0 support

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 5. Install Anomalib with extensions

Install Anomalib with VLM and CLIP support:

```bash
pip install anomalib[vlm,clip]
```

### 6. Install additional requirements

Install any additional dependencies required for benchmarking:

```bash
pip install -r additonal_requirments.txt
```

## Verification

To verify your installation is successful, you can check if PyTorch can detect your CUDA device:

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

## Next Steps

Once your environment is set up, you can proceed to run the benchmarking scripts from the main benchmark directory.