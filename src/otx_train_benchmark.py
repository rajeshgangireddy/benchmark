# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Training benchmark script for OTX (OpenVINO Training Extensions) models.

Uses OTX recipe files directly as configuration. All training parameters 
(model, epochs, batch_size, etc.) come from the recipe. Only benchmark-specific 
parameters (num_runs, seed, output_dir) are CLI arguments.

Usage:
  # Basic usage with OTX recipe
  python -m src.otx_train_benchmark --recipe /path/to/efficientnet_b0.yaml --data_root ./data

  # Multiple benchmark runs for statistics
  python -m src.otx_train_benchmark --recipe /path/to/recipe.yaml --data_root ./data --num_runs 5

  # Override device from recipe
  python -m src.otx_train_benchmark --recipe /path/to/recipe.yaml --data_root ./data --device cuda
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from src.benchmarkers.otx import OTXBenchmark
from src.utils.system_info import get_system_info
from src.utils.statistics import summarise_results, summarise_results_mlperf, flatten_system_info
from src.utils.metrics import export_detailed_metrics_to_excel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OTX Training Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - recipe provides all training config
  python -m src.otx_train_benchmark --recipe /path/to/efficientnet_b0.yaml --data_root ./data/flower_photos

  # Multiple benchmark runs for statistics
  python -m src.otx_train_benchmark --recipe /path/to/recipe.yaml --data_root ./data --num_runs 5

  # Override device from recipe
  python -m src.otx_train_benchmark --recipe /path/to/recipe.yaml --data_root ./data --device cuda
        """
    )

    # Required: OTX recipe file and data
    parser.add_argument("--recipe", type=str, required=True,
                        help="Path to OTX recipe YAML file (e.g., efficientnet_b0.yaml)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root directory")
    
    # Benchmark-specific parameters (not in OTX recipe)
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of training runs for statistics (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting seed for reproducibility (default: 42)")
    parser.add_argument("--wait_time", type=int, default=20,
                        help="Wait time between runs in seconds (default: 20)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Output directory for benchmark results (default: ./benchmark_results)")
    parser.add_argument("--export_otx_metrics", action="store_true",
                        help="Export detailed OTX training metrics to Excel")
    parser.add_argument("--barebones", action="store_true",
                        help="Disable logging, progress bars, checkpointing for minimal overhead")
    
    # Optional overrides for recipe values
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "xpu"],
                        help="Override device from recipe (cpu/cuda/xpu)")
    parser.add_argument("--work_dir", type=str, default="./otx-workspace",
                        help="OTX work directory (default: ./otx-workspace)")
    
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the OTX benchmarking workflow using recipe file."""
    
    # Validate recipe file exists
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe file not found: {recipe_path}")
        sys.exit(1)
    
    # Validate data root exists
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        sys.exit(1)
    
    # Determine device for system info
    device_for_sysinfo = None
    if args.device and args.device != "cpu":
        device_for_sysinfo = args.device  # 'cuda' or 'xpu'
    
    # Get system info
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Warning: Failed to retrieve full system info: {e}")
        system_info = {"error": str(e)}
    
    # Create benchmarker
    try:
        benchmarker = OTXBenchmark(args)
    except ImportError as e:
        print(f"Error: OTX is not installed. Install with: pip install otx")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to initialize OTX benchmarker: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Log configuration
    benchmarker.logger.info("=" * 60)
    benchmarker.logger.info("OTX Benchmark Configuration")
    benchmarker.logger.info("=" * 60)
    benchmarker.logger.info(f"Recipe: {recipe_path.absolute()}")
    benchmarker.logger.info(f"Data root: {data_root.absolute()}")
    benchmarker.logger.info(f"Num runs: {args.num_runs}")
    benchmarker.logger.info(f"Seed: {args.seed}")
    benchmarker.logger.info(f"Wait time: {args.wait_time}s")
    benchmarker.logger.info(f"Work dir: {args.work_dir}")
    if args.device:
        benchmarker.logger.info(f"Device override: {args.device}")
    if args.barebones:
        benchmarker.logger.info("Barebones mode: disabling logging, progress bars, checkpointing")
    benchmarker.logger.info("=" * 60)
    
    benchmarker.logger.info("System info:")
    for key, value in system_info.items():
        benchmarker.logger.info(f"  {key}: {value}")

    if args.num_runs < 3:
        benchmarker.logger.warning(f"num_runs={args.num_runs} < 3. MLPerf summary will be skipped.")

    # Run benchmark
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

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    recipe_name = recipe_path.stem
    device_str = args.device or "auto"
    filename = f"OTX_BM_{device_str}_{recipe_name}_runs-{args.num_runs}_seed-{args.seed}_{timestamp}.xlsx"
    result_path = output_dir / filename

    try:
        flattened_info = flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_info.items()), columns=["Component", "Details"])
        
        # Benchmark config info
        config_info = {
            "recipe": str(recipe_path.absolute()),
            "data_root": str(data_root.absolute()),
            "device_override": args.device or "(from recipe)",
            "num_runs": args.num_runs,
            "seed": args.seed,
            "wait_time": args.wait_time,
            "work_dir": args.work_dir,
        }
        config_df = pd.DataFrame(list(config_info.items()), columns=["Config", "Value"])

        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
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
        
        # Export OTX detailed metrics if requested
        if args.export_otx_metrics:
            export_detailed_metrics_to_excel(
                workspace_dir=args.work_dir,
                output_dir=output_dir,
                timestamp=timestamp,
                logger=benchmarker.logger
            )
            
    except Exception as e:
        benchmarker.logger.error(f"Error saving Excel: {e}")
        benchmarker.logger.error(traceback.format_exc())
        
        # Backup save as JSON
        try:
            backup_file = output_dir / f"otx_backup_results_{timestamp}.json"
            backup_data = {
                'timestamp': timestamp,
                'recipe': str(recipe_path),
                'data_root': str(data_root),
                'system_info': system_info,
                'results': results
            }
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            benchmarker.logger.info(f"Backup saved to {backup_file}")
        except Exception as backup_error:
            benchmarker.logger.error(f"Failed to save backup: {backup_error}")
        sys.exit(1)


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":    
    main()
