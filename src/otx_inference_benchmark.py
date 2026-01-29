# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Inference benchmark script for OTX (OpenVINO Training Extensions) models.
Wraps OTX's built-in engine.benchmark() with multi-run averaging and statistics.

Supports two modes:
1. Config file mode (recommended for reproducibility):
   python -m src.otx_inference_benchmark --config configs/inference_benchmark.yaml --checkpoint ./model.ckpt

2. CLI mode (quick testing):
   python -m src.otx_inference_benchmark --checkpoint ./model.ckpt --data_root ./data
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.system_info import get_system_info
from src.utils.statistics import summarise_inference_results, summarise_inference_results_mlperf, flatten_system_info
from src.utils.config import (
    BenchmarkConfig, 
    load_config, 
    save_config,
    merge_cli_args, 
    validate_inference_config,
    get_default_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OTX Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a config file (recommended for reproducibility)
  python -m src.otx_inference_benchmark --config configs/inference_benchmark.yaml --checkpoint ./model.ckpt

  # CLI-only mode
  python -m src.otx_inference_benchmark --checkpoint ./otx-workspace/best_checkpoint.ckpt --data_root ./data/coco

  # Override config with CLI arguments
  python -m src.otx_inference_benchmark --config configs/inference_benchmark.yaml --checkpoint ./model.ckpt --num_runs 5 --batch_size 4

  # Benchmark on CPU
  python -m src.otx_inference_benchmark --device cpu --checkpoint ./model.ckpt --data_root ./data
        """
    )

    # Config file (primary method)
    parser.add_argument("--config", type=str,
                        help="Path to YAML config file (recommended for reproducibility)")
    
    # These can override config file or be used standalone
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "xpu"],
                        help="Device to run inference on")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--data_root", type=str,
                        help="Path to dataset root directory")
    parser.add_argument("--num_runs", type=int,
                        help="Number of benchmark runs for statistics")
    parser.add_argument("--num_inferences", type=int,
                        help="Number of inference iterations per run")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for inference")
    parser.add_argument("--wait_time", type=int,
                        help="Wait time between runs in seconds")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory")
    parser.add_argument("--extended_stats", action="store_true",
                        help="Enable extended statistics from OTX benchmark")
    
    # Utility options
    parser.add_argument("--save_config", type=str,
                        help="Save the effective config to a YAML file and exit")
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """
    Build configuration from config file and/or CLI arguments.
    
    Priority: CLI args > config file > defaults
    """
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        config = get_default_config("inference")
    
    # Merge CLI arguments (they take precedence)
    config = merge_cli_args(config, args)
    
    return config


def run_otx_inference_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """
    Run OTX inference benchmark with multiple runs.
    
    Args:
        config: Benchmark configuration.
        
    Returns:
        List of result dictionaries from each run.
    """
    try:
        from otx.engine import Engine as OTXEngine
    except ImportError:
        print("Error: OTX is not installed. Install with: pip install otx")
        sys.exit(1)
    
    results = []
    device_map = {"cpu": "cpu", "cuda": "gpu", "xpu": "xpu"}
    otx_device = device_map.get(config.device.type, "auto")
    
    for run_idx in range(config.run.num_runs):
        print(f"\n{'=' * 50}")
        print(f"Inference benchmark run {run_idx + 1}/{config.run.num_runs}")
        print(f"{'=' * 50}")
        
        try:
            engine = OTXEngine(
                model=config.inference.checkpoint,
                data=config.data.root,
                device=otx_device,
            )
            
            # Run OTX's built-in benchmark
            benchmark_result = engine.benchmark(
                batch_size=config.inference.batch_size,
                n_iters=config.inference.num_inferences,
                extended_stats=config.inference.extended_stats,
                print_table=True,
            )
            
            # Parse results - OTX returns dict with string values
            result = {"run_id": run_idx + 1}
            
            # Latency (e.g., "0.0123 s" -> 0.0123)
            if "latency" in benchmark_result:
                latency_str = benchmark_result["latency"]
                latency_val = float(latency_str.replace(" s", "").replace("s", ""))
                result["latency_sec"] = latency_val
                result["avg_time"] = latency_val  # For compatibility with summarise functions
            
            # Throughput (e.g., "81.3 FPS" -> 81.3)
            if "throughput" in benchmark_result:
                throughput_str = benchmark_result["throughput"]
                throughput_val = float(throughput_str.replace(" FPS", "").replace("FPS", ""))
                result["throughput_fps"] = throughput_val
                result["fps"] = throughput_val  # For compatibility
            
            # Complexity
            if "complexity" in benchmark_result:
                result["complexity"] = benchmark_result["complexity"]
            
            results.append(result)
            print(f"Run {run_idx + 1} completed: latency={result.get('latency_sec', 'N/A')}s, "
                  f"throughput={result.get('throughput_fps', 'N/A')} FPS")
            
        except Exception as e:
            print(f"Error in run {run_idx + 1}: {e}")
            traceback.print_exc()
        
        if run_idx < config.run.num_runs - 1:
            print(f"Waiting {config.run.wait_time}s before next run...")
            time.sleep(config.run.wait_time)
    
    return results


def main():
    args = parse_args()
    
    # Build configuration
    config = build_config(args)
    
    # Handle --save_config
    if args.save_config:
        try:
            save_config(config, args.save_config)
            print(f"Config saved to: {args.save_config}")
        except Exception as e:
            print(f"Error saving config: {e}")
            sys.exit(1)
        sys.exit(0)
    
    # Validate configuration
    errors = validate_inference_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Set CUDA device from config
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device.cuda_visible_devices
    
    # Create output directory
    output_dir = Path(config.output.directory)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        sys.exit(1)
    
    # Get system info
    device_for_sysinfo = None if config.device.type == "cpu" else config.device.type
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Warning: Failed to retrieve system info: {e}")
        system_info = {}
    
    print("\n" + "=" * 50)
    print("OTX INFERENCE BENCHMARK")
    print("=" * 50)
    print(f"Checkpoint: {config.inference.checkpoint}")
    print(f"Device: {config.device.type}")
    print(f"Batch size: {config.inference.batch_size}")
    print(f"Iterations per run: {config.inference.num_inferences}")
    print(f"Number of runs: {config.run.num_runs}")
    print("=" * 50)
    
    # Run benchmarks
    results = run_otx_inference_benchmark(config)
    
    if not results:
        print("Error: No results collected. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted {len(results)}/{config.run.num_runs} runs successfully")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(config.inference.checkpoint).stem
    filename = f"OTX_INF_BM_{config.device.type}_{checkpoint_name}_runs-{config.run.num_runs}_{timestamp}.xlsx"
    result_path = output_dir / filename
    
    try:
        flattened_info = flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_info.items()), columns=["Component", "Details"])
        
        config_info = {
            "config_name": config.name or "CLI",
            "checkpoint": config.inference.checkpoint,
            "data_root": config.data.root,
            "device": config.device.type,
            "batch_size": config.inference.batch_size,
            "num_inferences": config.inference.num_inferences,
            "num_runs": config.run.num_runs,
            "extended_stats": config.inference.extended_stats,
        }
        config_df = pd.DataFrame(list(config_info.items()), columns=["Config", "Value"])

        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            config_df.to_excel(writer, sheet_name="Benchmark Config", index=False)
            
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name="Raw Results", index=False)

            summary = summarise_inference_results(results)
            summary_df = pd.DataFrame(summary, index=[0])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            if config.run.num_runs >= 3 and len(results) >= 3:
                mlperf_summary = summarise_inference_results_mlperf(results, config.run.num_runs)
                mlperf_df = pd.DataFrame(mlperf_summary, index=[0])
                mlperf_df.to_excel(writer, sheet_name="Summary MLPerf", index=False)

        print(f"\nResults saved to {result_path}")
        
        # Save config alongside results
        config_save_path = output_dir / f"inference_config_{timestamp}.yaml"
        save_config(config, config_save_path)
        print(f"Config saved to {config_save_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        if "mean_latency_sec" in summary:
            print(f"Mean Latency: {summary['mean_latency_sec']:.4f}s +/- {summary.get('std_latency_sec', 0):.4f}s")
        if "mean_throughput_fps" in summary:
            print(f"Mean Throughput: {summary['mean_throughput_fps']:.2f} +/- {summary.get('std_throughput_fps', 0):.2f} FPS")
        print("=" * 50)
            
    except Exception as e:
        print(f"Error saving Excel: {e}")
        traceback.print_exc()
        
        try:
            backup_file = output_dir / f"otx_inf_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'system_info': system_info,
                    'config': {
                        'checkpoint': config.inference.checkpoint,
                        'data_root': config.data.root,
                        'device': config.device.type,
                        'batch_size': config.inference.batch_size,
                        'num_inferences': config.inference.num_inferences,
                        'num_runs': config.run.num_runs,
                    },
                    'results': results
                }, f, indent=2, default=str)
            print(f"Backup saved to {backup_file}")
        except Exception as backup_error:
            print(f"Failed to save backup: {backup_error}")
        sys.exit(1)


if __name__ == "__main__":    
    main()
