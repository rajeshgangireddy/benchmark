# Setup Guide for NVIDIA Jetson AGX Orin Developer Kit

## Hardware Overview

This guide covers setting up the NVIDIA Jetson AGX Orin Developer Kit for running Anomalib benchmarks.

**Product Link:** [NVIDIA Jetson AGX Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

### Device Specifications

You can verify your device model by running:
```bash
cat /proc/device-tree/model
```
Expected output: `NVIDIA Jetson AGX Orin Developer Kit`
Some specs of this machine : 

| Component          | Specification                                                 |
| ------------------ | ------------------------------------------------------------- |
| AI Performance     | 275 TOPS                                                      |
| GPU                | 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores |
| GPU Max Frequency  | 1.3 GHz                                                       |
| CPU                | 12-core Arm Cortex-A78AE v8.2 64-bit CPU (3MB L2 + 6MB L3)   |
| CPU Max Frequency  | 2.2 GHz                                                       |
| DL Accelerator     | 2x NVDLA v2                                                   |
| Memory             | 64GB 256-bit LPDDR5 (204.8GB/s bandwidth)                     |
| Storage            | 64GB eMMC 5.1                                                 |
| Power              | 15W - 60W (configurable)                                      |


##  Why Docker ? 

On the Jetson platform, PyTorch can be installed either through the system-wide JetPack SDK or via Docker. We chose Docker because it's easier to set up and makes the environment reproducible across different systems.

The JetPack SDK includes system-wide CUDA and cuDNN installations managed by NVIDIA, but Docker provides better isolation and version control for development work.

## Prerequisites

Before starting, make sure your Jetson has the following installed:

### 1. Docker Engine
Check your Docker version:
```bash
docker --version
```
Expected: `Docker version 28.5.1` or newer

### 2. NVIDIA GPU Drivers
Verify CUDA is available:
```bash
nvidia-smi
nvcc --version
```
Expected output should show Driver Version 540.4.0 or newer and CUDA Version 12.6 or newer.

### 3. NVIDIA Container Toolkit
Check if the toolkit is installed:
```bash
dpkg -l | grep nvidia-container-toolkit
```
You should see `nvidia-container-toolkit` listed with version 1.17.8 or newer.

## Environment Setup

### Step 1: Pull the Official PyTorch Container

We use NVIDIA's official PyTorch container from the NGC catalog. The container we're using is version 25.10-py3-igpu, which includes PyTorch 2.9.0a0 and CUDA 13.0.2. The "igpu" variant is specifically designed for Jetson devices.

```bash
sudo docker pull nvcr.io/nvidia/pytorch:25.10-py3-igpu
```

More information about this release: [PyTorch Release 25.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-10.html)

### Step 2: Launch the Container

Run the container with the necessary flags to enable GPU access and mount your workspace:

```bash
sudo docker run -it --rm \
  --runtime nvidia \
  --ipc=host \
  --network host \
  -v ~/jetson_workspace:/workspace \
  nvcr.io/nvidia/pytorch:25.10-py3-igpu \
  /bin/bash
```

**What each flag does:**
- `-it` - Runs the container in interactive mode with a terminal
- `--rm` - Automatically removes the container when you exit
- `--runtime nvidia` - Bridges the Jetson's GPU hardware with the Docker container
- `--ipc=host` - Enables shared memory for better multiprocessing performance (important for PyTorch data loaders)
- `--network host` - Uses the host's network stack, which simplifies networking setup
- `-v ~/jetson_workspace:/workspace` - Mounts your local directory into the container so files persist after the container stops

### Step 3: Verify GPU Access

Once inside the container, verify that PyTorch can access the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
Expected output: `True`

```bash
python3 -c "import torch; print(torch.version.cuda)"
```
Expected output: `13.0` or similar

### Step 4: Install Dependencies

The container comes with a pre-release version of PyTorch. We need to replace it with the stable release and install additional dependencies.

#### Install OpenCV dependency
```bash
apt update
apt install -y libgl1
```

#### Install stable PyTorch
The container includes PyTorch 2.9.0a0, which is not a standard release. Replace it with the stable version:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### Install Anomalib and benchmarking tools
```bash
pip install anomalib[clip,vlm]
pip install openpyxl
pip install pandas
pip install psutil
```

The `openpyxl` package is required for saving benchmark results to Excel files.

### Step 5: Save the Container Image

After installing all dependencies, save your configured container so you don't have to repeat these steps.

In a new shell, find the container ID:
```bash
sudo docker ps -a
```

Commit the container to create a new image:
```bash
sudo docker commit <container_id> anomalib:latest
```

Now you can start your configured environment anytime with:
```bash
sudo docker run -it --runtime nvidia --ipc=host --network host \
  -v ~/jetson_workspace:/workspace \
  anomalib:latest \
  /bin/bash
```

## Power Configuration

For accurate benchmarking, the Jetson should run at maximum performance with all CPUs enabled.

Check the current power mode (in native shell, not docker shell):
```bash
sudo nvpmodel --query
```

Expected output:
```
NV Power Mode: MAXN
0
```

If you see a different mode, set it to MAXN:
```bash
sudo nvpmodel -m 0
```

This ensures all CPU cores are active and the device runs at full performance.

## Running Benchmarks

Once inside your configured container, you can run benchmarks. For example:

```bash
python /workspace/benchmark_script.py \
  --device gpu \
  --category transistor \
  --epochs 1 \
  --models <model_names> \
  --num-runs 5 \
  --output-dir /workspace/results
```

## Troubleshooting

### If you need to set a proxy
If you're behind a corporate proxy, set these environment variables inside the container:
```bash
export http_proxy=<corportate_proxy:port> 
export https_proxy=<corportate_proxy:port>
```

### If the system clock is wrong
If you get errors while using apt or 
```bash
sudo date -s "2025-11-10 14:30:00"
```
Replace with the current date and time.

### Alternative container sources
If you need different PyTorch versions or configurations, check out the [Jetson Containers](https://github.com/dusty-nv/jetson-containers) project, which maintains community-built containers for various frameworks.

## Additional Notes

The JetPack SDK includes the Jetson Linux Driver Package with bootloader, Linux kernel, Ubuntu desktop environment, and a complete set of libraries for GPU computing, multimedia, graphics, and computer vision acceleration. However, using Docker on top of JetPack gives you more flexibility in managing Python environments and dependencies without affecting the system-wide installation.

## apex error

If you get apex error, then please do : 
```
git clone https://github.com/ptrblck/apex.git
cd apex
git checkout apex_no_distributed
pip install -v --no-cache-dir ./
```
