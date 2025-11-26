# Intel XPU Environment Setup for Anomalib Benchmarking

This guide will help you set up an Intel XPU-enabled environment for running Anomalib benchmarks on Intel GPUs.

**Note:** Detailed hardware requirements and driver installation instructions are available in the official [PyTorch Intel XPU Getting Started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) guide.

## Prerequisites

- Intel GPU with XPU support
- Intel GPU drivers installed (follow instructions from the PyTorch documentation linked above)
- Conda package manager installed

## Driver Verification

After installing the Intel GPU drivers, verify the installation:

```bash
clinfo | grep "Device Name"
```

You should see your Intel GPU device listed. If you don't see any devices, ensure your user has render group permissions:

```bash
sudo gpasswd -a ${USER} render
newgrp render
```

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

### 3. Install PyTorch with Intel XPU support

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

### 4. Install Anomalib with extensions

Install Anomalib with VLM and CLIP support:

```bash
pip install anomalib[vlm,clip]
```

### 5. Install additional requirements

Install any additional dependencies required for benchmarking:

```bash
pip install -r additonal_requirements.txt
```

## Verification

To verify your installation is successful, check if PyTorch can detect your Intel XPU device:

```python
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU device: {torch.xpu.get_device_name(0) if torch.xpu.is_available() else "None"}')"
```

## Next Steps

Once your environment is set up, you can proceed to run the benchmarking scripts from the main benchmark directory.
