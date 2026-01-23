# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Inference benchmark script for OTX (OpenVINO Training Extensions) models.
Wraps OTX's built-in engine.benchmark() with multi-run averaging and statistics.
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

from utils.system_info import get_system_info
from utils.statistics import summarise_inference_results, summarise_inference_results_mlperf, flatten_system_info

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OTX Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark inference with a trained checkpoint
  python otx_inference_benchmark.py --checkpoint ./otx-workspace/best_checkpoint.ckpt --data_root ./data/coco

  # Benchmark with multiple runs and custom batch size
  python otx_inference_benchmark.py --checkpoint ./model.ckpt --data_root ./data --num_runs 5 --batch_size 4

  # Benchmark on CPU
  python otx_inference_benchmark.py --device cpu --checkpoint ./model.ckpt --data_root ./data
        """
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "xpu"],
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--num_runs", type=int, default=3, 
                        help="Number of benchmark runs for statistics (default: 3)")
    parser.add_argument("--num_inferences", type=int, default=100,
                        help="Number of inference iterations per run (default: 100)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--wait_time", type=int, default=10, 
                        help="Wait time between runs in seconds (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", 
                        help="Output directory (default: ./benchmark_results)")
    parser.add_argument("--extended_stats", action="store_true",
                        help="Enable extended statistics from OTX benchmark")
    
    return parser.parse_args()


def run_otx_inference_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    Run OTX inference benchmark with multiple runs.
    
    Args:
        args: Parsed command line arguments.
        
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
    otx_device = device_map.get(args.device, "auto")
    
    for run_idx in range(args.num_runs):
        print(f"\n{'=' * 50}")
        print(f"Inference benchmark run {run_idx + 1}/{args.num_runs}")
        print(f"{'=' * 50}")
        
        try:
            engine = OTXEngine(
                model=args.checkpoint,
                data=args.data_root,
                device=otx_device,
            )
            
            # Run OTX's built-in benchmark
            benchmark_result = engine.benchmark(
                batch_size=args.batch_size,
                n_iters=args.num_inferences,
                extended_stats=args.extended_stats,
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
        
        if run_idx < args.num_runs - 1:
            print(f"Waiting {args.wait_time}s before next run...")
            time.sleep(args.wait_time)
    
    return results


def main():
    args = parse_args()
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        sys.exit(1)
    
    # Get system info
    device_for_sysinfo = None if args.device == "cpu" else ("cuda" if args.device == "cuda" else "xpu")
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Warning: Failed to retrieve system info: {e}")
        system_info = {}
    
    print("\n" + "=" * 50)
    print("OTX INFERENCE BENCHMARK")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations per run: {args.num_inferences}")
    print(f"Number of runs: {args.num_runs}")
    print("=" * 50)
    
    # Run benchmarks
    results = run_otx_inference_benchmark(args)
    
    if not results:
        print("Error: No results collected. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted {len(results)}/{args.num_runs} runs successfully")
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    filename = f"OTX_INF_BM_{args.device}_{checkpoint_name}_runs-{args.num_runs}_{timestamp}.xlsx"
    result_path = output_dir / filename
    
    try:
        flattened_info = flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_info.items()), columns=["Component", "Details"])
        
        config_info = {
            "checkpoint": args.checkpoint,
            "data_root": args.data_root,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_inferences": args.num_inferences,
            "num_runs": args.num_runs,
        }
        config_df = pd.DataFrame(list(config_info.items()), columns=["Config", "Value"])

        with pd.ExcelWriter(result_path) as writer:
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            config_df.to_excel(writer, sheet_name="Benchmark Config", index=False)
            
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name="Raw Results", index=False)

            summary = summarise_inference_results(results)
            summary_df = pd.DataFrame(summary, index=[0])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            if args.num_runs >= 3 and len(results) >= 3:
                mlperf_summary = summarise_inference_results_mlperf(results, args.num_runs)
                mlperf_df = pd.DataFrame(mlperf_summary, index=[0])
                mlperf_df.to_excel(writer, sheet_name="Summary MLPerf", index=False)

        print(f"\nResults saved to {result_path}")
        
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
                    'system_info': system_info,
                    'config': vars(args),
                    'results': results
                }, f, indent=2, default=str)
            print(f"Backup saved to {backup_file}")
        except Exception as backup_error:
            print(f"Failed to save backup: {backup_error}")
        sys.exit(1)


if __name__ == "__main__":    
    main()
