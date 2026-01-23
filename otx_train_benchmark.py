# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Training benchmark script for OTX (OpenVINO Training Extensions) models.
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

from otx_benchmarker import OTXBenchmark
from utils.system_info import get_system_info
from utils.statistics import summarise_results, summarise_results_mlperf, flatten_system_info
from utils.dataset import OTX_TASK_TYPES, list_otx_models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OTX Training Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a detection model
  python otx_train_benchmark.py --device cuda --task DETECTION --model atss_mobilenetv2 --data_root ./data/coco

  # Benchmark a classification model  
  python otx_train_benchmark.py --device xpu --task MULTI_CLASS_CLS --model efficientnet_b0 --data_root ./data/imagenet

  # Use a recipe file directly
  python otx_train_benchmark.py --device cuda --task DETECTION --model ./recipes/detection/yolox.yaml --data_root ./data

  # List available models for a task
  python otx_train_benchmark.py --list_models --task DETECTION
        """
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "xpu"],
                        help="Device to run benchmark on (default: cuda)")
    parser.add_argument("--task", type=str, choices=OTX_TASK_TYPES,
                        help="OTX task type")
    parser.add_argument("--model", type=str,
                        help="Model name (e.g., atss_mobilenetv2) or path to recipe YAML")
    parser.add_argument("--data_root", type=str,
                        help="Path to dataset root directory")
    parser.add_argument("--num_runs", type=int, default=5, 
                        help="Number of training runs (default: 5)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Starting seed (default: 42)")
    parser.add_argument("--wait_time", type=int, default=20, 
                        help="Wait time between runs in seconds (default: 20)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", 
                        help="Output directory (default: ./benchmark_results)")
    parser.add_argument("--max_epochs", type=int, default=10, 
                        help="Maximum training epochs (default: 10)")
    parser.add_argument("--precision", type=str, default=None, 
                        choices=["16", "32", "bf16-mixed"],
                        help="Training precision (default: framework default)")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, 
                        help="Evaluation batch size (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Data loading workers (default: 4)")
    
    # Utility options
    parser.add_argument("--list_models", action="store_true",
                        help="List available models for the specified task and exit")
    
    args = parser.parse_args()
    
    # Handle --list_models
    if args.list_models:
        if not args.task:
            parser.error("--list_models requires --task to be specified")
        try:
            models = list_otx_models(task=args.task)
            print(f"\nAvailable models for task '{args.task}':")
            for model in models:
                print(f"  - {model}")
            print(f"\nTotal: {len(models)} models")
        except ImportError as e:
            print(f"Error: {e}")
        sys.exit(0)
    
    # Validate required args for benchmarking
    if not args.task:
        parser.error("--task is required for benchmarking")
    if not args.model:
        parser.error("--model is required for benchmarking")
    if not args.data_root:
        parser.error("--data_root is required for benchmarking")
    
    return args


def benchmark(args: argparse.Namespace) -> None:
    """Run the OTX benchmarking workflow."""
    device = args.device
    
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        sys.exit(1)
    
    device_for_sysinfo = None if device == "cpu" else ("cuda" if device == "cuda" else "xpu")
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Error: Failed to retrieve system info: {e}")
        sys.exit(1)
    
    try:
        benchmarker = OTXBenchmark(args)
    except ImportError as e:
        print(f"Error: OTX is not installed. Install with: pip install otx")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to initialize OTX benchmarker: {e}")
        sys.exit(1)
        
    benchmarker.logger.info("System info:")
    for key, value in system_info.items():
        benchmarker.logger.info(f"  {key}: {value}")

    benchmarker.logger.info(f"OTX Benchmark Configuration:")
    benchmarker.logger.info(f"  Task: {args.task}")
    benchmarker.logger.info(f"  Model: {args.model}")
    benchmarker.logger.info(f"  Device: {args.device}")
    benchmarker.logger.info(f"  Data root: {args.data_root}")
    benchmarker.logger.info(f"  Max epochs: {args.max_epochs}")
    benchmarker.logger.info(f"  Num runs: {args.num_runs}")

    if args.num_runs < 3:
        benchmarker.logger.warning(f"num_runs={args.num_runs} < 3. MLPerf summary will be skipped.")

    try:
        results = benchmarker.run_benchmark()
    except Exception as e:
        benchmarker.logger.error(f"Critical error during benchmark: {e}")
        benchmarker.logger.error(traceback.format_exc())
        sys.exit(1)
    
    if not results:
        benchmarker.logger.error("No results collected. Exiting.")
        sys.exit(1)
    
    benchmarker.logger.info(f"Completed {len(results)}/{args.num_runs} runs successfully")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = Path(args.model).stem if args.model.endswith(('.yaml', '.yml')) else args.model
    filename = (f"OTX_BM_{args.device}_{args.task}_{model_name}_"
                f"runs-{args.num_runs}_seed-{args.seed}_{timestamp}.xlsx")
    result_path = output_dir / filename

    try:
        flattened_info = flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_info.items()), columns=["Component", "Details"])
        
        # Add benchmark config to system info
        config_info = {
            "task": args.task,
            "model": args.model,
            "data_root": args.data_root,
            "max_epochs": args.max_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "precision": args.precision or "default",
        }
        config_df = pd.DataFrame(list(config_info.items()), columns=["Config", "Value"])

        with pd.ExcelWriter(result_path) as writer:
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            config_df.to_excel(writer, sheet_name="Benchmark Config", index=False)
            
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name="Raw Results", index=False)

            summary = summarise_results(results)
            summary_df = pd.DataFrame(summary, index=[0])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            if args.num_runs >= 3 and len(results) >= 3:
                mlperf_summary = summarise_results_mlperf(results)
                mlperf_df = pd.DataFrame(mlperf_summary, index=[0])
                mlperf_df.to_excel(writer, sheet_name="Summary MLPerf", index=False)
            else:
                benchmarker.logger.warning("Skipping MLPerf summary (requires >= 3 runs)")

        benchmarker.logger.info(f"Results saved to {result_path}")
        benchmarker.logger.info(f"Summary: {summary}")
        if args.num_runs >= 3 and len(results) >= 3:
            benchmarker.logger.info(f"MLPerf Summary: {mlperf_summary}")
            
    except Exception as e:
        benchmarker.logger.error(f"Error saving Excel: {e}")
        benchmarker.logger.error(traceback.format_exc())
        
        try:
            backup_file = output_dir / f"otx_backup_results_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump({
                    'system_info': system_info,
                    'config': vars(args),
                    'results': results
                }, f, indent=2, default=str)
            benchmarker.logger.info(f"Backup saved to {backup_file}")
        except Exception as backup_error:
            benchmarker.logger.error(f"Failed to save backup: {backup_error}")
        sys.exit(1)


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":    
    main()
