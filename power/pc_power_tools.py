#!/usr/bin/env python3
"""
Unified Power Analysis and Benchmark Runner for PC (CUDA/XPU/CPU)

This script provides:
1. Power log analysis with statistics and visualization
2. Automated benchmark running with power monitoring
3. Support for CUDA (NVIDIA), XPU (Intel), and CPU devices

Usage:
    # Analyze existing power log
    python pc_power_tools.py analyze power_log.csv [--plot]
    
    # Run benchmark with power monitoring
    python pc_power_tools.py benchmark --device cuda --num_runs 5 [benchmark args]

Requirements:
    - pandas
    - psutil
    - pynvml (for NVIDIA GPUs)
    - matplotlib (optional, for plots)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def analyze_power_log(csv_file: Path, generate_plots: bool = False, output_file: Optional[Path] = None):
    """Analyze power consumption data from CSV log."""
    
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    # Load data
    print(f"Loading power data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Calculate duration
    if 'timestamp' in df.columns and len(df) > 1:
        duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    else:
        duration = len(df)  # Estimate based on sample count
    
    num_samples = len(df)
    
    # Prepare statistics
    stats = {
        'duration_seconds': duration,
        'num_samples': num_samples,
        'components': {}
    }
    
    # Analyze each power component
    power_columns = [col for col in df.columns if col.endswith('_W')]
    
    for col in power_columns:
        component_name = col.replace('_power_W', '').replace('_W', '').upper()
        stats['components'][component_name] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
        }
    
    # Calculate total energy
    if 'total_power_W' in df.columns:
        mean_power_W = df['total_power_W'].mean()
        stats['energy_J'] = mean_power_W * duration
        stats['energy_Wh'] = stats['energy_J'] / 3600
    
    # Print report
    print_report(stats)
    
    # Save report if requested
    if output_file:
        report_text = generate_report_text(stats)
        output_file.write_text(report_text)
        print(f"\nReport saved to: {output_file}")
    
    # Generate plots if requested
    if generate_plots:
        try:
            import matplotlib.pyplot as plt
            
            # Create time axis
            if 'timestamp' in df.columns:
                df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            else:
                df['time_s'] = range(len(df))
            
            # Determine number of subplots needed
            num_components = len(power_columns)
            if num_components == 0:
                print("No power data columns found for plotting.")
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Total power over time
            ax1 = axes[0]
            if 'total_power_W' in df.columns:
                ax1.plot(df['time_s'], df['total_power_W'], label='Total Power', alpha=0.8, linewidth=1)
                ax1.fill_between(df['time_s'], df['total_power_W'], alpha=0.3)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Power (W)')
            ax1.set_title('Total Power Consumption Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Component breakdown
            ax2 = axes[1]
            for col in power_columns:
                if col != 'total_power_W':
                    component_name = col.replace('_power_W', '').replace('_W', '').upper()
                    ax2.plot(df['time_s'], df[col], label=component_name, alpha=0.8, linewidth=1)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Power (W)')
            ax2.set_title('Power Consumption by Component')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = csv_file.stem + '_plot.png'
            plt.savefig(plot_file, dpi=150)
            print(f"Plot saved to: {plot_file}")
            
            # Histogram
            if 'total_power_W' in df.columns:
                fig2, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['total_power_W'], bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Power (W)')
                ax.set_ylabel('Frequency')
                ax.set_title('Power Consumption Distribution')
                ax.axvline(df['total_power_W'].mean(), color='red', linestyle='--', 
                          label=f"Mean: {df['total_power_W'].mean():.2f} W")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                hist_file = csv_file.stem + '_histogram.png'
                plt.savefig(hist_file, dpi=150)
                print(f"Histogram saved to: {hist_file}")
            
        except ImportError:
            print("\nWarning: matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\nError generating plots: {e}")
            import traceback
            traceback.print_exc()


def print_report(stats: Dict):
    """Print formatted power consumption report."""
    print("\n" + "=" * 70)
    print("PC POWER CONSUMPTION ANALYSIS")
    print("=" * 70)
    print(f"\nDuration: {stats['duration_seconds']:.2f} seconds ({stats['duration_seconds']/60:.2f} minutes)")
    print(f"Samples:  {stats['num_samples']}")
    
    print("\n" + "-" * 70)
    print("POWER CONSUMPTION BY COMPONENT")
    print("-" * 70)
    
    for component, data in stats['components'].items():
        print(f"\n{component}:")
        print(f"  Mean:   {data['mean']:7.3f} W")
        print(f"  Median: {data['median']:7.3f} W")
        print(f"  Std:    {data['std']:7.3f} W")
        print(f"  Range:  {data['min']:7.3f} - {data['max']:7.3f} W")
    
    if 'energy_J' in stats:
        print("\n" + "-" * 70)
        print("TOTAL ENERGY CONSUMPTION")
        print("-" * 70)
        print(f"\n{stats['energy_J']:.2f} J ({stats['energy_Wh']:.6f} Wh)")
    
    print("=" * 70)


def generate_report_text(stats: Dict) -> str:
    """Generate report text for file output."""
    lines = []
    lines.append("=" * 70)
    lines.append("PC POWER CONSUMPTION ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"\nDuration: {stats['duration_seconds']:.2f} seconds")
    lines.append(f"Samples:  {stats['num_samples']}")
    
    for component, data in stats['components'].items():
        lines.append(f"\n{component}: Mean={data['mean']:.3f}W, Max={data['max']:.3f}W")
    
    if 'energy_J' in stats:
        lines.append(f"\nTotal Energy: {stats['energy_J']:.2f} J ({stats['energy_Wh']:.6f} Wh)")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def run_benchmark_with_power_monitoring(device: str, benchmark_args: list):
    """Run benchmark with automatic power monitoring."""
    
    script_dir = Path(__file__).parent
    power_script = script_dir / "pc_power_monitor.py"
    benchmark_script = script_dir / "benchmark_script.py"
    
    # Check scripts exist
    if not power_script.exists():
        print(f"Error: {power_script} not found")
        sys.exit(1)
    if not benchmark_script.exists():
        print(f"Error: {benchmark_script} not found")
        sys.exit(1)
    
    # Generate timestamp-based output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    power_log = f"pc_power_{device}_{timestamp}.csv"
    output_dir = f"benchmark_results_{timestamp}"
    
    print("=" * 70)
    print(f"BENCHMARK WITH POWER MONITORING ({device.upper()})")
    print("=" * 70)
    print(f"Timestamp:    {timestamp}")
    print(f"Device:       {device.upper()}")
    print(f"Power Log:    {power_log}")
    print(f"Output Dir:   {output_dir}")
    print("=" * 70)
    print()
    
    # Start power monitoring
    print("Starting power monitoring...")
    power_process = subprocess.Popen(
        [sys.executable, str(power_script), '--device', device, '--output', power_log, '--interval', '1.0'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    time.sleep(3)  # Let power monitoring initialize
    
    if power_process.poll() is not None:
        print("Error: Power monitoring failed to start")
        output = power_process.stdout.read() if power_process.stdout else ""
        print(output)
        sys.exit(1)
    
    print(f"Power monitoring started (PID: {power_process.pid})")
    print()
    
    # Prepare benchmark command
    benchmark_cmd = [sys.executable, str(benchmark_script), "--output_dir", output_dir] + benchmark_args
    
    print("Starting benchmark...")
    print(f"Command: {' '.join(benchmark_cmd)}")
    print()
    
    # Run benchmark
    try:
        result = subprocess.run(benchmark_cmd)
        benchmark_exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        benchmark_exit_code = 130
    
    # Stop power monitoring
    print("\n\nStopping power monitoring...")
    power_process.send_signal(signal.SIGINT)
    try:
        power_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        power_process.kill()
        power_process.wait()
    
    print("\n" + "=" * 70)
    print("GENERATING POWER ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze power data
    power_log_path = Path(power_log)
    if power_log_path.exists():
        power_report = Path(output_dir) / "power_analysis.txt"
        analyze_power_log(power_log_path, generate_plots=True, output_file=power_report)
        
        # Move files to output directory
        power_log_path.rename(Path(output_dir) / power_log)
        
        # Move plot files
        for plot_file in Path('.').glob(f"{power_log_path.stem}*.png"):
            plot_file.rename(Path(output_dir) / plot_file.name)
    else:
        print(f"Warning: Power log not found at {power_log_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Benchmark exit code: {benchmark_exit_code}")
    print(f"All results saved to: {output_dir}/")
    print("=" * 70)
    
    sys.exit(benchmark_exit_code)


def main():
    parser = argparse.ArgumentParser(
        description="PC Power Analysis and Benchmark Tools (CUDA/XPU/CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze power log
  %(prog)s analyze power_log.csv --plot
  
  # Run benchmark with power monitoring (CUDA)
  %(prog)s benchmark --device cuda --num_runs 5 --max_epochs 20
  
  # Run benchmark with power monitoring (XPU)
  %(prog)s benchmark --device xpu --num_runs 3 --max_epochs 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze power log')
    analyze_parser.add_argument('csv_file', type=Path, help='Power log CSV file')
    analyze_parser.add_argument('--plot', action='store_true', help='Generate plots')
    analyze_parser.add_argument('--output', type=Path, help='Save report to file')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark with power monitoring')
    benchmark_parser.add_argument('--device', type=str, required=True, 
                                   choices=['cuda', 'xpu', 'cpu'],
                                   help='Device to monitor (cuda/xpu/cpu)')
    benchmark_parser.add_argument('benchmark_args', nargs=argparse.REMAINDER, 
                                   help='Arguments to pass to benchmark_script.py')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_power_log(args.csv_file, args.plot, args.output)
    elif args.command == 'benchmark':
        run_benchmark_with_power_monitoring(args.device, args.benchmark_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
