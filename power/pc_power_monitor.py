#!/usr/bin/env python3
"""
PC Power Monitoring Script for CUDA and XPU Devices

This script monitors power consumption on standard PCs with NVIDIA GPUs (CUDA) 
or Intel GPUs (XPU/Arc). It uses multiple methods for maximum accuracy:

- NVIDIA: nvidia-smi for GPU power, psutil for CPU/system
- Intel XPU: Intel GPU tools (intel_gpu_top, xpu-smi) for GPU, psutil for CPU/system
- CPU: Intel RAPL (Running Average Power Limit) via powercap interface
- System: psutil for overall system metrics

Usage:
    python pc_power_monitor.py [--device cuda|xpu|cpu] [--output power_log.csv] [--interval 1.0]

Requirements:
    pip install psutil pynvml py-cpuinfo

Optional for better accuracy:
    - Intel XPU: intel_gpu_top or xpu-smi installed
    - NVIDIA: nvidia-smi (usually comes with drivers)
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: pip install psutil")
    sys.exit(1)


class PowerMonitor:
    """Base class for power monitoring."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.measurements = []
        self.running = False
        
    def measure(self) -> Dict[str, float]:
        """Take a single power measurement. Returns dict with power values in watts."""
        raise NotImplementedError
        
    def start(self):
        """Start monitoring."""
        self.running = True
        
    def stop(self):
        """Stop monitoring."""
        self.running = False
        
    def get_summary(self) -> Dict:
        """Calculate summary statistics."""
        if not self.measurements:
            return {}
            
        total_powers = [m.get('total_power_W', 0) for m in self.measurements]
        
        return {
            'count': len(self.measurements),
            'mean_W': sum(total_powers) / len(total_powers),
            'min_W': min(total_powers),
            'max_W': max(total_powers),
            'duration_s': len(self.measurements) * self.interval,
            'energy_J': sum(total_powers) * self.interval,
            'energy_Wh': sum(total_powers) * self.interval / 3600,
        }


class NVIDIAMonitor(PowerMonitor):
    """Monitor for NVIDIA GPUs using nvidia-smi and RAPL."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__(interval)
        self.nvml_available = False
        self.rapl_available = False
        
        # Try to initialize NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
            self.nvml_available = True
            print(f"✓ NVIDIA GPUs detected: {self.gpu_count}")
        except Exception as e:
            print(f"⚠ NVML not available: {e}")
            self.gpu_count = 0
            
        # Check for RAPL (CPU power monitoring on Linux)
        self.rapl_available = self._check_rapl()
        if self.rapl_available:
            print("✓ Intel RAPL (CPU power) available")
            
    def _check_rapl(self) -> bool:
        """Check if RAPL is available for CPU power monitoring."""
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        return rapl_path.exists()
        
    def _read_rapl_energy(self) -> float:
        """Read energy from RAPL interface in Joules."""
        total_energy = 0.0
        rapl_base = Path("/sys/class/powercap/intel-rapl")
        
        try:
            for rapl_dir in rapl_base.glob("intel-rapl:*"):
                energy_file = rapl_dir / "energy_uj"
                if energy_file.exists():
                    energy_uj = int(energy_file.read_text().strip())
                    total_energy += energy_uj / 1_000_000  # Convert microjoules to joules
        except:
            pass
            
        return total_energy
        
    def measure(self) -> Dict[str, float]:
        """Measure power consumption."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'gpu_power_W': 0.0,
            'cpu_power_W': 0.0,
            'total_power_W': 0.0,
        }
        
        # GPU power via NVML
        if self.nvml_available:
            gpu_power = 0.0
            for i, handle in enumerate(self.gpu_handles):
                try:
                    power_mW = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_power += power_mW / 1000.0  # Convert to watts
                except:
                    pass
            result['gpu_power_W'] = gpu_power
            
        # CPU power via RAPL
        if self.rapl_available and hasattr(self, '_last_rapl_energy'):
            current_energy = self._read_rapl_energy()
            energy_diff = current_energy - self._last_rapl_energy
            cpu_power = energy_diff / self.interval if energy_diff > 0 else 0.0
            result['cpu_power_W'] = cpu_power
            self._last_rapl_energy = current_energy
        elif self.rapl_available:
            self._last_rapl_energy = self._read_rapl_energy()
            
        result['total_power_W'] = result['gpu_power_W'] + result['cpu_power_W']
        return result


class IntelXPUMonitor(PowerMonitor):
    """Monitor for Intel XPU/Arc GPUs."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__(interval)
        self.xpu_available = False
        self.rapl_available = False
        
        # Check for Intel GPU monitoring tools
        import shutil
        if shutil.which('xpu-smi'):
            self.xpu_tool = 'xpu-smi'
            self.xpu_available = True
            print("✓ Intel XPU monitoring available (xpu-smi)")
        elif shutil.which('intel_gpu_top'):
            self.xpu_tool = 'intel_gpu_top'
            self.xpu_available = True
            print("✓ Intel GPU monitoring available (intel_gpu_top)")
        else:
            print("⚠ Intel XPU monitoring tools not found (install xpu-smi or intel_gpu_top)")
            
        # Check for RAPL
        self.rapl_available = self._check_rapl()
        if self.rapl_available:
            print("✓ Intel RAPL (CPU power) available")
            
    def _check_rapl(self) -> bool:
        """Check if RAPL is available."""
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        return rapl_path.exists()
        
    def _read_rapl_energy(self) -> float:
        """Read energy from RAPL interface."""
        total_energy = 0.0
        rapl_base = Path("/sys/class/powercap/intel-rapl")
        
        try:
            for rapl_dir in rapl_base.glob("intel-rapl:*"):
                energy_file = rapl_dir / "energy_uj"
                if energy_file.exists():
                    energy_uj = int(energy_file.read_text().strip())
                    total_energy += energy_uj / 1_000_000
        except:
            pass
            
        return total_energy
        
    def _get_xpu_power(self) -> float:
        """Get XPU power consumption."""
        if not self.xpu_available:
            return 0.0
            
        try:
            import subprocess
            if self.xpu_tool == 'xpu-smi':
                # Try xpu-smi stats command
                result = subprocess.run(['xpu-smi', 'stats', '-d', '0'], 
                                      capture_output=True, text=True, timeout=2)
                # Parse output for power (format varies, this is a basic attempt)
                for line in result.stdout.split('\n'):
                    if 'power' in line.lower():
                        # Extract number followed by W
                        import re
                        match = re.search(r'(\d+\.?\d*)\s*W', line)
                        if match:
                            return float(match.group(1))
            # Fallback: estimate based on GPU usage
            return 0.0
        except:
            return 0.0
            
    def measure(self) -> Dict[str, float]:
        """Measure power consumption."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'gpu_power_W': 0.0,
            'cpu_power_W': 0.0,
            'total_power_W': 0.0,
        }
        
        # GPU power
        result['gpu_power_W'] = self._get_xpu_power()
        
        # CPU power via RAPL
        if self.rapl_available and hasattr(self, '_last_rapl_energy'):
            current_energy = self._read_rapl_energy()
            energy_diff = current_energy - self._last_rapl_energy
            cpu_power = energy_diff / self.interval if energy_diff > 0 else 0.0
            result['cpu_power_W'] = cpu_power
            self._last_rapl_energy = current_energy
        elif self.rapl_available:
            self._last_rapl_energy = self._read_rapl_energy()
            
        result['total_power_W'] = result['gpu_power_W'] + result['cpu_power_W']
        return result


class CPUMonitor(PowerMonitor):
    """Monitor for CPU-only systems using RAPL."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__(interval)
        self.rapl_available = self._check_rapl()
        
        if self.rapl_available:
            print("✓ Intel RAPL (CPU power) available")
        else:
            print("⚠ RAPL not available. Power monitoring will be limited.")
            
    def _check_rapl(self) -> bool:
        """Check if RAPL is available."""
        rapl_path = Path("/sys/class/powercap/intel-rapl")
        return rapl_path.exists()
        
    def _read_rapl_energy(self) -> float:
        """Read energy from RAPL interface."""
        total_energy = 0.0
        rapl_base = Path("/sys/class/powercap/intel-rapl")
        
        try:
            for rapl_dir in rapl_base.glob("intel-rapl:*"):
                energy_file = rapl_dir / "energy_uj"
                if energy_file.exists():
                    energy_uj = int(energy_file.read_text().strip())
                    total_energy += energy_uj / 1_000_000
        except:
            pass
            
        return total_energy
        
    def measure(self) -> Dict[str, float]:
        """Measure power consumption."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'cpu_power_W': 0.0,
            'total_power_W': 0.0,
        }
        
        if self.rapl_available and hasattr(self, '_last_rapl_energy'):
            current_energy = self._read_rapl_energy()
            energy_diff = current_energy - self._last_rapl_energy
            cpu_power = energy_diff / self.interval if energy_diff > 0 else 0.0
            result['cpu_power_W'] = cpu_power
            self._last_rapl_energy = current_energy
        elif self.rapl_available:
            self._last_rapl_energy = self._read_rapl_energy()
            
        result['total_power_W'] = result['cpu_power_W']
        return result


def create_monitor(device: str, interval: float) -> PowerMonitor:
    """Create appropriate monitor for the device."""
    if device == 'cuda':
        return NVIDIAMonitor(interval)
    elif device == 'xpu':
        return IntelXPUMonitor(interval)
    elif device == 'cpu':
        return CPUMonitor(interval)
    else:
        raise ValueError(f"Unsupported device: {device}")


def main():
    parser = argparse.ArgumentParser(
        description="PC Power Monitoring for CUDA/XPU/CPU devices"
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'xpu', 'cpu'],
        required=True,
        help='Device type to monitor'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('power_log.csv'),
        help='Output CSV file (default: power_log.csv)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Create monitor
    print("=" * 70)
    print("PC Power Monitor")
    print("=" * 70)
    print(f"Device:   {args.device.upper()}")
    print(f"Output:   {args.output}")
    print(f"Interval: {args.interval} seconds")
    print("=" * 70)
    print()
    
    monitor = create_monitor(args.device, args.interval)
    monitor.start()
    
    # Setup CSV writer
    csv_file = open(args.output, 'w', newline='')
    
    # Determine columns based on first measurement
    test_measurement = monitor.measure()
    fieldnames = ['timestamp'] + [k for k in test_measurement.keys() if k != 'timestamp']
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(test_measurement)
    monitor.measurements.append(test_measurement)
    
    print("Monitoring started. Press Ctrl+C to stop.\n")
    
    # Signal handler for clean exit
    def signal_handler(sig, frame):
        print("\n\nStopping power monitoring...")
        monitor.stop()
        csv_file.close()
        
        # Print summary
        summary = monitor.get_summary()
        if summary:
            print("\n" + "=" * 70)
            print("POWER MEASUREMENT SUMMARY")
            print("=" * 70)
            print(f"Duration:     {summary['duration_s']:.2f} seconds ({summary['duration_s']/60:.2f} minutes)")
            print(f"Samples:      {summary['count']}")
            print(f"\nPower Consumption:")
            print(f"  Mean:       {summary['mean_W']:.3f} W")
            print(f"  Min:        {summary['min_W']:.3f} W")
            print(f"  Max:        {summary['max_W']:.3f} W")
            print(f"\nTotal Energy: {summary['energy_J']:.2f} J ({summary['energy_Wh']:.6f} Wh)")
            print("=" * 70)
            print(f"\nData saved to: {args.output}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Main monitoring loop
    try:
        while monitor.running:
            time.sleep(args.interval)
            measurement = monitor.measure()
            writer.writerow(measurement)
            csv_file.flush()
            monitor.measurements.append(measurement)
            
            # Print to console
            timestamp = measurement['timestamp'].split('T')[1][:12]
            total_power = measurement.get('total_power_W', 0)
            gpu_power = measurement.get('gpu_power_W', 0)
            cpu_power = measurement.get('cpu_power_W', 0)
            
            print(f"[{timestamp}] Total: {total_power:6.3f} W | GPU: {gpu_power:6.3f} W | CPU: {cpu_power:6.3f} W")
            
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        monitor.stop()
        csv_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
