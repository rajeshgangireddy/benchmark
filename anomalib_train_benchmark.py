# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Training benchmark script for Anomalib models.
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

from anomalib_benchmarker import AnomalibBenchmark
from utils.system_info import get_system_info
from utils.statistics import summarise_results, summarise_results_mlperf, flatten_system_info

# Expected versions for validation
EXPECTED_VERSIONS = {
    "torch": "2.9",
    "python": "3.12.13",
    "anomalib": "2.2.0",
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_versions(system_info: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Check current versions against expected versions."""
    mismatches = {}
    version_map = {
        "torch": system_info.get("torch_version", "Unknown"),
        "python": system_info.get("python_version", "Unknown"),
        "anomalib": system_info.get("anomalib_version", "Unknown"),
    }
    for component, expected in EXPECTED_VERSIONS.items():
        current = version_map.get(component, "Unknown")
        if current != expected:
            mismatches[component] = (expected, current)
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomalib Training Benchmark")

    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda", "xpu"],
                        help="Device to run benchmark on")
    parser.add_argument("--num_runs", type=int, default=5, 
                        help="Number of training+testing runs (default: 5)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Starting seed for reproducibility (default: 42)")
    parser.add_argument("--wait_time", type=int, default=20, 
                        help="Wait time between runs in seconds (default: 20)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", 
                        help="Output directory for results (default: ./benchmark_results)")
    parser.add_argument("--max_epochs", type=int, default=20, 
                        help="Maximum training epochs (default: 20)")
    parser.add_argument("--barebones", action="store_true",
                        help="Disable logging, progress bars, checkpointing for minimal overhead")
    parser.add_argument("--precision", type=str, default=None, choices=["16", "32", "bf16-mixed"],
                        help="Training precision (default: framework default)")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--eval_batch_size", type=int, default=32, 
                        help="Evaluation batch size (default: 32)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Data loading workers (default: 8)")
    parser.add_argument("--category", type=str, required=True,
                        help="MVTecAD category (e.g., bottle, transistor)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Anomalib model name (e.g., Padim, Patchcore)")
    
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> None:
    """Run the benchmarking workflow."""
    device = args.device
    
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        sys.exit(1)
    
    device_for_sysinfo = None if device == "cpu" else device
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Error: Failed to retrieve system info: {e}")
        sys.exit(1)
    
    try:
        benchmarker = AnomalibBenchmark(args)
    except Exception as e:
        print(f"Error: Failed to initialize benchmarker: {e}")
        sys.exit(1)
        
    benchmarker.logger.info("System info:")
    for key, value in system_info.items():
        benchmarker.logger.info(f"  {key}: {value}")

    version_mismatches = check_versions(system_info)
    for component, (expected, current) in version_mismatches.items():
        benchmarker.logger.warning(f"Version mismatch {component}: expected {expected}, got {current}")

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
    filename = (f"BM_{args.device}_{args.model_name}_{args.category}_"
                f"runs-{args.num_runs}_seed-{args.seed}_{timestamp}.xlsx")
    result_path = output_dir / filename

    try:
        flattened_info = flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_info.items()), columns=["Component", "Details"])

        with pd.ExcelWriter(result_path) as writer:
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            
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
            backup_file = output_dir / f"backup_results_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump({'system_info': system_info, 'results': results}, f, indent=2)
            benchmarker.logger.info(f"Backup saved to {backup_file}")
        except Exception as backup_error:
            benchmarker.logger.error(f"Failed to save backup: {backup_error}")
        sys.exit(1)


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":    
    main()
