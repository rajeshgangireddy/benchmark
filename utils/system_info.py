# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
All utility functions for hardware (CPU, GPU, etc) information.
Provides comprehensive system and device information for benchmarking.
"""
import platform
import subprocess
from datetime import datetime
from typing import Any

import psutil
import torch


# Constants
GB_TO_BYTES = 1024 ** 3
MB_TO_BYTES = 1024 ** 2
KB_TO_BYTES = 1024


# ============================================================================
# PUBLIC API
# ============================================================================

def get_system_info(device: str | None = None) -> dict[str, Any]:
    """Get comprehensive system and device information for benchmarking.

    Args:
        device: Optional device type ('cuda' or 'xpu'). If None, returns only CPU and system info.

    Returns:
        Dictionary with system information, CPU details, and device-specific info if requested.
        
    Raises:
        ValueError: If device is not None, 'cuda', or 'xpu'.
        RuntimeError: If requested device is not available.
    """
    # Validate device parameter
    if device is not None and device not in ["cuda", "xpu"]:
        raise ValueError(f"Invalid device: {device}. Must be None, 'cuda', or 'xpu'.")
    
    # Get general system information
    info = _get_general_system_info()
    
    # Always include CPU information
    info["cpu"] = _get_cpu_info()
    
    # Add device-specific information if requested
    if device == "cuda":
        info["device"] = _get_cuda_info()
    elif device == "xpu":
        info["device"] = _get_xpu_info()
    
    return info


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _safe_int_conversion(value: str) -> int | None:
    """Safely convert string to int, return None if invalid."""
    try:
        return int(value) if value != "N/A" else None
    except (ValueError, TypeError):
        return None


def _safe_float_conversion(value: str) -> float | None:
    """Safely convert string to float, return None if invalid."""
    try:
        return float(value) if value != "N/A" else None
    except (ValueError, TypeError):
        return None


def _run_subprocess_command(command: list[str], timeout: int = 10) -> str | None:
    """Run subprocess command and return output, None on failure."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _parse_proc_cpuinfo() -> dict[str, Any]:
    """Parse /proc/cpuinfo and extract CPU information in a single pass."""
    cpu_details = {}
    
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            
        for line in cpuinfo.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
                
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            
            if key == "model name" and "name" not in cpu_details:
                cpu_details["name"] = value
            elif key == "vendor_id" and "vendor" not in cpu_details:
                cpu_details["vendor"] = value
            elif key == "flags" and "features" not in cpu_details:
                flags = value.split()
                cpu_features = []
                for feature in ["avx", "avx2", "avx512f", "sse4_1", "sse4_2"]:
                    if feature in flags:
                        cpu_features.append(feature.upper())
                cpu_details["features"] = cpu_features
                
    except Exception:
        pass  # Fall back to basic info if parsing fails
        
    return cpu_details


def _get_software_info() -> dict[str, Any]:
    """Get software library versions (anomalib, torch, etc)."""
    software_info = {}
    for lib in ["torch", "anomalib"]:
        try:
            module = __import__(lib)
            software_info[f"{lib}_version"] = module.__version__
        except ImportError:
            software_info[f"{lib}_version"] = "Not installed"
        except Exception:
            software_info[f"{lib}_version"] = "Unknown"
    return software_info        


def _get_general_system_info() -> dict[str, Any]:
    """Get general system information common to all devices."""
    system_info = {
        "timestamp": datetime.now().isoformat() + "Z",
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "total_system_ram_gb": round(psutil.virtual_memory().total / GB_TO_BYTES, 2),
    }
    system_info.update(_get_software_info())
    return system_info


def _get_cpu_info() -> dict[str, Any]:
    """Get detailed CPU information."""
    cpu_info = {
        "name": platform.processor(),
        "architecture": platform.machine(),
        "core_count": psutil.cpu_count(logical=False),
        "thread_count": psutil.cpu_count(logical=True),
        "load_percent": round(psutil.cpu_percent(interval=1), 2),
    }
    
    # Get CPU frequency information
    try:
        freq_info = psutil.cpu_freq()
        if freq_info:
            cpu_info["max_frequency_ghz"] = round(freq_info.max / 1000, 2) if freq_info.max else None
            cpu_info["current_frequency_ghz"] = round(freq_info.current / 1000, 2) if freq_info.current else None
    except Exception:
        cpu_info["max_frequency_ghz"] = None
        cpu_info["current_frequency_ghz"] = None
    
    # Get detailed CPU information from /proc/cpuinfo on Linux
    if platform.system() == "Linux":
        cpu_info.update(_parse_proc_cpuinfo())
    
    # Get cache sizes using lscpu
    lscpu_output = _run_subprocess_command(["lscpu"])
    if lscpu_output:
        for line in lscpu_output.split("\n"):
            if "L1d cache:" in line:
                cpu_info["l1_cache_size"] = line.split(":")[1].strip()
            elif "L2 cache:" in line:
                cpu_info["l2_cache_size"] = line.split(":")[1].strip()
            elif "L3 cache:" in line:
                cpu_info["l3_cache_size"] = line.split(":")[1].strip()
    
    return cpu_info


def _get_cuda_info() -> dict[str, Any]:
    """Get detailed CUDA device information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found.")
    
    if device_count > 1:
        import warnings
        warnings.warn(
            f"Multiple CUDA devices detected ({device_count}). Using GPU 0 for benchmarking. "
            "For consistent results, ensure only one GPU is visible.",
            UserWarning,
            stacklevel=2
        )
    
    device_properties = torch.cuda.get_device_properties(0)
    
    cuda_info = {
        "type": "cuda",
        "device_count": device_count,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{device_properties.major}.{device_properties.minor}",
        "total_memory_gb": round(device_properties.total_memory / GB_TO_BYTES, 2),
        "cuda_runtime_version": torch.version.cuda,
    }
    
    # Add device properties
    safe_attrs = {
        "multiprocessors": "multi_processor_count",
        "max_threads_per_block": "max_threads_per_block",
        "max_threads_per_multiprocessor": "max_threads_per_multi_processor",
        "l2_cache_size_mb": "l2_cache_size",
        "is_integrated": "is_integrated",
        "max_shared_memory_per_block": "max_shared_memory_per_block",
        "warp_size": "warp_size",
    }
    
    for key, attr_name in safe_attrs.items():
        value = getattr(device_properties, attr_name, None)
        if value is not None:
            if key == "l2_cache_size_mb":
                cuda_info[key] = round(value / MB_TO_BYTES, 2)
            elif key == "max_shared_memory_per_block":
                cuda_info[key] = round(value / KB_TO_BYTES, 2)
            else:
                cuda_info[key] = value
    
    # Get memory information
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        cuda_info["free_memory_gb"] = round(free_mem / GB_TO_BYTES, 2)
        cuda_info["used_memory_gb"] = round((total_mem - free_mem) / GB_TO_BYTES, 2)
    except Exception:
        pass
    
    # Get driver version using nvidia-smi
    driver_output = _run_subprocess_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]
    )
    if driver_output:
        cuda_info["driver_version"] = driver_output.split('\n')[0]
    
    # Get additional GPU stats
    gpu_stats_output = _run_subprocess_command([
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.memory,clocks.current.sm",
        "--format=csv,noheader,nounits"
    ])
    
    if gpu_stats_output:
        values = gpu_stats_output.split(', ')
        if len(values) >= 6:
            cuda_info["memory_used_mb"] = _safe_int_conversion(values[0])
            cuda_info["gpu_utilization_percent"] = _safe_int_conversion(values[1])
            cuda_info["temperature_c"] = _safe_int_conversion(values[2])
            cuda_info["power_draw_w"] = _safe_float_conversion(values[3])
            cuda_info["memory_clock_mhz"] = _safe_int_conversion(values[4])
            cuda_info["sm_clock_mhz"] = _safe_int_conversion(values[5])
    
    return cuda_info


def _get_xpu_info() -> dict[str, Any]:
    """Get detailed XPU (Intel GPU) device information."""
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is not available on this system.")
    
    device_count = torch.xpu.device_count()
    if device_count == 0:
        raise RuntimeError("No XPU devices found.")
    
    if device_count > 1:
        raise RuntimeError("Multiple XPU devices found. Please run on a single XPU for benchmarking.")
    
    device_properties = torch.xpu.get_device_properties(0)
    
    xpu_info = {
        "type": "xpu",
        "device_count": device_count,
        "device_name": torch.xpu.get_device_name(0),
        "total_memory_gb": round(device_properties.total_memory / GB_TO_BYTES, 2),
        "max_compute_units": getattr(device_properties, 'max_compute_units', None),
        "max_work_group_size": getattr(device_properties, 'max_work_group_size', None),
        "is_integrated": getattr(device_properties, 'is_integrated', None),
    }
    
    # Get XPU runtime version
    try:
        xpu_info["runtime_version"] = torch.xpu.version
    except Exception:
        xpu_info["runtime_version"] = "Unknown"
    
    # Get driver information using sycl-ls
    sycl_output = _run_subprocess_command(["sycl-ls"])
    if sycl_output:
        xpu_info["sycl_devices"] = sycl_output
    
    return xpu_info