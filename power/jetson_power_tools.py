#!/usr/bin/env python3
"""
Jetson Power Analysis and Benchmark Runner

This script provides two main functions:
1. Analyze power consumption logs from jetson_power_monitor.sh
2. Run benchmarks with automatic power monitoring

Usage:
    # Analyze existing power log
    python jetson_power_tools.py analyze power_log.csv [--plot]
    
    # Run benchmark with power monitoring
    python jetson_power_tools.py benchmark --device cuda --num_runs 5 [other benchmark args]

Requirements:
    - pandas (required)
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Calculate statistics
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    num_samples = len(df)
    
    stats = {
        'duration_seconds': duration,
        'num_samples': num_samples,
        'total_current': {
            'mean': df['total_current_mW'].mean(),
            'std': df['total_current_mW'].std(),
            'min': df['total_current_mW'].min(),
            'max': df['total_current_mW'].max(),
            'median': df['total_current_mW'].median(),
        },
        'total_avg': {
            'mean': df['total_avg_mW'].mean(),
            'std': df['total_avg_mW'].std(),
            'min': df['total_avg_mW'].min(),
            'max': df['total_avg_mW'].max(),
            'median': df['total_avg_mW'].median(),
        },
    }
    
    # Per-rail statistics
    rails = [
        ('gpu_soc', 'GPU/SOC'),
        ('cpu_cv', 'CPU/CV'),
        ('sys_5v', 'System 5V')
    ]
    
    stats['rails'] = {}
    for rail_key, rail_name in rails:
        current_col = f'{rail_key}_current_mW'
        avg_col = f'{rail_key}_avg_mW'
        if current_col in df.columns:
            stats['rails'][rail_name] = {
                'current_mean': df[current_col].mean(),
                'current_max': df[current_col].max(),
                'avg_mean': df[avg_col].mean(),
                'avg_max': df[avg_col].max(),
            }
    
    # Energy calculation
    mean_power_W = stats['total_current']['mean'] / 1000
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
            
            df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Total power over time
            ax1 = axes[0]
            ax1.plot(df['time_s'], df['total_current_mW'], label='Current', alpha=0.7, linewidth=0.8)
            ax1.plot(df['time_s'], df['total_avg_mW'], label='Average', alpha=0.7, linewidth=0.8)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Power (mW)')
            ax1.set_title('Total Power Consumption')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Per-rail breakdown
            ax2 = axes[1]
            ax2.plot(df['time_s'], df['gpu_soc_current_mW'], label='GPU/SOC', alpha=0.7, linewidth=0.8)
            ax2.plot(df['time_s'], df['cpu_cv_current_mW'], label='CPU/CV', alpha=0.7, linewidth=0.8)
            ax2.plot(df['time_s'], df['sys_5v_current_mW'], label='System 5V', alpha=0.7, linewidth=0.8)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Power (mW)')
            ax2.set_title('Per-Rail Power Consumption')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = csv_file.stem + '_plot.png'
            plt.savefig(plot_file, dpi=150)
            print(f"Plot saved to: {plot_file}")
            
        except ImportError:
            print("\nWarning: matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\nError generating plots: {e}")


def print_report(stats: Dict):
    """Print formatted power consumption report."""
    print("\n" + "=" * 70)
    print("JETSON POWER CONSUMPTION ANALYSIS")
    print("=" * 70)
    print(f"\nDuration: {stats['duration_seconds']:.2f} seconds ({stats['duration_seconds']/60:.2f} minutes)")
    print(f"Samples:  {stats['num_samples']}")
    
    print("\n" + "-" * 70)
    print("TOTAL POWER (All Rails Combined)")
    print("-" * 70)
    
    curr = stats['total_current']
    print(f"\nCurrent Power:")
    print(f"  Mean:   {curr['mean']:7.1f} mW ({curr['mean']/1000:6.3f} W)")
    print(f"  Median: {curr['median']:7.1f} mW ({curr['median']/1000:6.3f} W)")
    print(f"  Std:    {curr['std']:7.1f} mW ({curr['std']/1000:6.3f} W)")
    print(f"  Range:  {curr['min']:7.1f} - {curr['max']:7.1f} mW")
    
    avg = stats['total_avg']
    print(f"\nAverage Power:")
    print(f"  Mean:   {avg['mean']:7.1f} mW ({avg['mean']/1000:6.3f} W)")
    print(f"  Median: {avg['median']:7.1f} mW ({avg['median']/1000:6.3f} W)")
    print(f"  Range:  {avg['min']:7.1f} - {avg['max']:7.1f} mW")
    
    print("\n" + "-" * 70)
    print("PER-RAIL BREAKDOWN")
    print("-" * 70)
    
    for rail_name, data in stats['rails'].items():
        print(f"\n{rail_name}:")
        print(f"  Current Mean: {data['current_mean']:7.1f} mW ({data['current_mean']/1000:6.3f} W)")
        print(f"  Current Max:  {data['current_max']:7.1f} mW ({data['current_max']/1000:6.3f} W)")
    
    print("\n" + "-" * 70)
    print("ENERGY CONSUMPTION")
    print("-" * 70)
    print(f"\nTotal Energy: {stats['energy_J']:.2f} J ({stats['energy_Wh']:.6f} Wh)")
    print("=" * 70)


def generate_report_text(stats: Dict) -> str:
    """Generate report text for file output."""
    lines = []
    lines.append("=" * 70)
    lines.append("JETSON POWER CONSUMPTION ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"\nDuration: {stats['duration_seconds']:.2f} seconds")
    lines.append(f"Samples:  {stats['num_samples']}")
    lines.append(f"\nMean Power: {stats['total_current']['mean']:.1f} mW ({stats['total_current']['mean']/1000:.3f} W)")
    lines.append(f"Max Power:  {stats['total_current']['max']:.1f} mW ({stats['total_current']['max']/1000:.3f} W)")
    lines.append(f"\nTotal Energy: {stats['energy_J']:.2f} J ({stats['energy_Wh']:.6f} Wh)")
    lines.append("=" * 70)
    return "\n".join(lines)


def run_benchmark_with_power_monitoring(benchmark_args: list):
    """Run benchmark with automatic power monitoring."""
    
    script_dir = Path(__file__).parent
    power_script = script_dir / "jetson_power_monitor.sh"
    benchmark_script = script_dir.parent / "benchmark_script.py"
    
    # Check scripts exist
    if not power_script.exists():
        print(f"Error: {power_script} not found")
        sys.exit(1)
    if not benchmark_script.exists():
        print(f"Error: {benchmark_script} not found")
        sys.exit(1)
    
    # Make power script executable
    power_script.chmod(0o755)
    
    # Generate timestamp-based output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    power_log = f"jetson_power_{timestamp}.csv"
    output_dir = f"benchmark_results_{timestamp}"
    
    print("=" * 70)
    print("BENCHMARK WITH POWER MONITORING")
    print("=" * 70)
    print(f"Timestamp:    {timestamp}")
    print(f"Power Log:    {power_log}")
    print(f"Output Dir:   {output_dir}")
    print("=" * 70)
    print()
    
    # Start power monitoring
    print("Starting power monitoring...")
    power_process = subprocess.Popen(
        [str(power_script), power_log, "1000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    time.sleep(2)  # Let power monitoring initialize
    
    if power_process.poll() is not None:
        print("Error: Power monitoring failed to start")
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
        power_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        power_process.kill()
    
    print("\n" + "=" * 70)
    print("GENERATING POWER ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze power data
    if Path(power_log).exists():
        power_report = Path(output_dir) / "power_analysis.txt"
        analyze_power_log(Path(power_log), generate_plots=True, output_file=power_report)
        
        # Move files to output directory
        Path(power_log).rename(Path(output_dir) / power_log)
        
        plot_file = Path(f"{Path(power_log).stem}_plot.png")
        if plot_file.exists():
            plot_file.rename(Path(output_dir) / plot_file.name)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Benchmark exit code: {benchmark_exit_code}")
    print(f"All results saved to: {output_dir}/")
    print("=" * 70)
    
    sys.exit(benchmark_exit_code)


def main():
    parser = argparse.ArgumentParser(
        description="Jetson Power Analysis and Benchmark Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze power log
  %(prog)s analyze power_log.csv --plot
  
  # Run benchmark with power monitoring
  %(prog)s benchmark --device cuda --num_runs 5 --max_epochs 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze power log')
    analyze_parser.add_argument('csv_file', type=Path, help='Power log CSV file')
    analyze_parser.add_argument('--plot', action='store_true', help='Generate plots')
    analyze_parser.add_argument('--output', type=Path, help='Save report to file')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark with power monitoring',
                                              add_help=False)
    
    # Parse known args to allow passing through benchmark arguments
    args, unknown = parser.parse_known_args()
    
    if args.command == 'analyze':
        # Re-parse with full analyze parser to get help and validation
        args = parser.parse_args()
        analyze_power_log(args.csv_file, args.plot, args.output)
    elif args.command == 'benchmark':
        # Pass all unknown arguments to benchmark
        run_benchmark_with_power_monitoring(unknown)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
