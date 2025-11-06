import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarker import AnomalibBenchmark
from utils.system_info import get_system_info

# Expected Versions
TORCH_VERSION = "2.9"
PYTHON_VERSION = "3.12.13"
ANOMALIB_VERSION = "2.2.0"

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only one GPU is visible for benchmarking

def main():
    """
    Main entry point for the Anomalib benchmarking script.
    Parses command-line arguments and executes the benchmark.
    """
    parser = argparse.ArgumentParser(description="Anomalib Benchmarking Script")

    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda", "xpu"],
                        help="Device to run the benchmark on.")
    parser.add_argument("--num_runs", type=int, default=5, 
                        help="Number of times the model training+testing will be run. Default is 5.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Starting seed for reproducibility. Each run will increment the seed by 1. Default is 42.")
    parser.add_argument("--wait_time", type=int, default=20, 
                        help="Time to wait before starting a new run. Default is 20 seconds.")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", 
                        help="Directory to save the benchmark results and any other info. Default is './benchmark_results'.")

    parser.add_argument("--max_epochs", type=int, default=20, 
                        help="Maximum number of epochs for training for models that support. Default is 20.")
    parser.add_argument("--barebones", action="store_true",
                        help="Flag to enable barebones mode which disables logging, progress bars, and checkpointing for minimal overhead during benchmarking.")

    # Engine, Data, Model Related Arguments
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Training batch size. Default is 32 .")
    parser.add_argument("--eval_batch_size", type=int, default=32, 
                        help="Evaluation batch size. Default is 32 .")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of workers for data loading. Default is 8 .")
    parser.add_argument("--category", type=str, required=True,
                        help="MVTecAD dataset category to use for benchmarking. E.g., 'bottle', 'capsule', etc.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Anomalib model name to benchmark. E.g., 'padim', 'stfpm', etc.")
    args = parser.parse_args()

    # Validate num_runs for MLPerf summary (needs at least 3 runs)
    if args.num_runs < 3:
        print(f"Warning: num_runs={args.num_runs} is less than 3. MLPerf-style summary requires at least 3 runs.")
        print("MLPerf summary will be skipped in the results.")
    
    benchmark(args)

def summarise_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarises the benchmark results by calculating the mean and standard deviation 
    of training and testing times, and averaging the metrics across all runs.

    Args:
        results (list[dict[str, Any]]): List of benchmark results from each run.

    Returns:
        dict[str, Any]: Summarised benchmark results including means and standard deviations.
    """
    summarised_result = {}
    num_runs = len(results)
    
    # Training and testing times
    training_times = [result["training_time_sec"] for result in results]
    testing_times = [result["testing_time_sec"] for result in results]
    
    summarised_result["mean_training_time_sec"] = statistics.mean(training_times)
    summarised_result["std_training_time_sec"] = statistics.stdev(training_times) if num_runs > 1 else 0.0
    
    summarised_result["mean_testing_time_sec"] = statistics.mean(testing_times)
    summarised_result["std_testing_time_sec"] = statistics.stdev(testing_times) if num_runs > 1 else 0.0
    

    # Average metrics with standard deviation
    metric_keys = [key for key in results[0].keys() if key not in ("run_id", "seed", "training_time_sec", "testing_time_sec")]
    for key in metric_keys:
        metric_values = [result[key] for result in results]
        summarised_result[f"mean_{key}"] = statistics.mean(metric_values)
        summarised_result[f"std_{key}"] = statistics.stdev(metric_values) if num_runs > 1 else 0.0

    return summarised_result

def summarise_results_mlperf(results: list[dict[str, Any]], num_runs: int) -> dict[str, Any]:
    """
    Summarises the benchmark results by dropping the fastest and slowest training times,
    and calculating the arithmetic mean of the remaining runs.

    Args:
        results (list[dict[str, Any]]): List of benchmark results from each run.
        num_runs (int): Total number of runs.

    Returns:
        dict[str, Any]: Summarised benchmark results.
        
    Raises:
        ValueError: If num_runs < 3 (MLPerf method requires at least 3 runs).
    """
    if num_runs < 3:
        raise ValueError(f"MLPerf summary requires at least 3 runs, but only {num_runs} provided.")
    
    if len(results) < 3:
        raise ValueError(f"MLPerf summary requires at least 3 completed runs, but only {len(results)} available.")

    training_times = [result["training_time_sec"] for result in results]

    slowest_run_idx = training_times.index(max(training_times))
    fastest_run_idx = training_times.index(min(training_times))

    filtered_results = [result for idx, result in enumerate(results) 
                        if idx not in (slowest_run_idx, fastest_run_idx)]
    summarised_result = summarise_results(filtered_results)
    return summarised_result
    

def _flatten_system_info(system_info: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested dictionaries in system_info into separate entries with prefixes.
    
    This improves Excel readability by converting complex nested structures like:
    {'cpu': {'name': 'Intel i9', 'cores': 12}} 
    into:
    {'cpu': '', 'cpu_name': 'Intel i9', 'cpu_cores': 12}
    
    The section headers (like 'cpu') are added as empty entries for visual grouping.
    
    Args:
        system_info: Dictionary containing system information with potentially nested structures
        
    Returns:
        Dictionary with flattened structure including section headers for Excel display
    """
    flattened = {}
    
    for key, value in system_info.items():
        if isinstance(value, dict):
            # Add section header for the complex object
            flattened[key] = "--- Section Header ---"
            
            # For nested dictionaries, flatten with prefix
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        else:
            # Keep simple values as-is
            flattened[key] = value
    
    return flattened


def check_versions(system_info: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """
    Checks if the current versions of torch, python, and anomalib match the expected versions.
    Logs a warning if there is a mismatch.

    Args:
        system_info (dict[str, Any]): Dictionary containing system information.
        
    Returns:
        dict[str, tuple[str, str]]: Dictionary mapping component names to (expected, actual) version tuples.
    """
    current_torch_version = system_info.get("PyTorch Version", "Unknown")
    current_python_version = system_info.get("Python Version", "Unknown")
    current_anomalib_version = system_info.get("Anomalib Version", "Unknown")

    mismatches = {}
    if current_torch_version != TORCH_VERSION:
        mismatches["PyTorch Version"] = (TORCH_VERSION, current_torch_version)
    if current_python_version != PYTHON_VERSION:
        mismatches["Python Version"] = (PYTHON_VERSION, current_python_version)
    if current_anomalib_version != ANOMALIB_VERSION:
        mismatches["Anomalib Version"] = (ANOMALIB_VERSION, current_anomalib_version)

    return mismatches

def benchmark(args: argparse.Namespace) -> None:
    """
    Runs the benchmarking process for an Anomalib model on the specified device.
    
    This function orchestrates the entire benchmarking workflow:
    1. Collects system and device information
    2. Executes multiple training/testing runs
    3. Computes summary statistics (standard and MLPerf-style)
    4. Saves results to an Excel file
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - device: Target device ('cpu', 'cuda', or 'xpu')
            - num_runs: Number of benchmark runs to execute
            - seed: Starting random seed
            - model_name: Name of the Anomalib model
            - category: MVTecAD dataset category
            - output_dir: Directory to save results
            - Other training/data configuration parameters
    
    Raises:
        SystemExit: If critical errors occur during benchmarking or file operations.
    
    Example:
        >>> args = parser.parse_args(['--device', 'cuda', '--num_runs', '5'])
        >>> benchmark(args)
    """
    
    device = args.device
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)
    
    # Get hardware and software info with error handling
    # For CPU, pass None to get_system_info; for cuda/xpu, pass the device type
    device_for_sysinfo = None if device == "cpu" else device
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Error: Failed to retrieve system information: {e}")
        sys.exit(1)
    
    # Initialize benchmarker
    try:
        benchmarker = AnomalibBenchmark(args)
    except Exception as e:
        print(f"Error: Failed to initialize benchmarker: {e}")
        sys.exit(1)
        
    benchmarker.logger.info("Starting benchmark with the following system info:")
    for key, value in system_info.items():
        benchmarker.logger.info(f"{key}: {value}")

    benchmarker.logger.info("Starting benchmark...")

    version_mismatches = check_versions(system_info)
    if version_mismatches:
        for component, (expected, current) in version_mismatches.items():
            benchmarker.logger.warning(f"Version mismatch for {component}: Expected {expected}, but got {current}.")

    try:
        results = benchmarker.run_benchmark()
    except Exception as e:
        benchmarker.logger.error(f"Critical error during benchmark execution: {e}")
        benchmarker.logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Check if we have any results
    if not results:
        benchmarker.logger.error("No benchmark results were collected. Exiting.")
        sys.exit(1)
    
    benchmarker.logger.info(f"Completed {len(results)}/{args.num_runs} runs successfully.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = (f"BM_{args.device}_{args.model_name}_"
                f"{args.category}_"
                f"runs-{args.num_runs}_"
                f"seed-{args.seed}_"
                f"{timestamp}.xlsx")

    result_excel_file_path = output_dir / filename

    # Save all data into an excel file
    # Sheet 1 : System Info
    # Sheet 2 : Benchmark Results Raw
    # Sheet 3 : Benchmark Results Summary Processed 
    
    try:
        # Create flattened system info for better Excel readability
        flattened_system_info = _flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_system_info.items()), columns=["Component", "Details"])

        with pd.ExcelWriter(result_excel_file_path) as writer:
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            # Sheet 2 : Benchmark Results Raw
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name="Benchmark Results Raw", index=False)

            # Sheet 3 : Benchmark Results Summary UnProcessed
            summarised_result = summarise_results(results)
            summarised_result_df = pd.DataFrame(summarised_result, index=[0])
            summarised_result_df.to_excel(writer, sheet_name="Summary", index=False)

            # Sheet 4 : Benchmark Results Summary Processed (MLPerf style)
            # We follow MLperf way of summary : MLPerf drops the fastest and slowest times, 
            # reporting the arithmetic mean of the remaining runs as the result.
            if args.num_runs >= 3 and len(results) >= 3:
                mlperf_summarised_result = summarise_results_mlperf(results, args.num_runs)
                mlperf_summarised_result_df = pd.DataFrame(mlperf_summarised_result, index=[0])
                mlperf_summarised_result_df.to_excel(writer, sheet_name="Summary MLPERF", index=False)
            else:
                benchmarker.logger.warning(f"Skipping MLPerf summary: requires at least 3 runs, got {len(results)}.")

        benchmarker.logger.info(f"Benchmark results saved to {result_excel_file_path}.")
        benchmarker.logger.info(f"Summarised Results: {summarised_result}")
        if args.num_runs >= 3 and len(results) >= 3:
            benchmarker.logger.info(f"MLPerf Summarised Results: {mlperf_summarised_result}")
            
    except Exception as e:
        benchmarker.logger.error(f"Error saving results to Excel file: {e}")
        
        benchmarker.logger.error(traceback.format_exc())
        # Save raw results as backup
        try:
            backup_file = output_dir / f"backup_results_{timestamp}.json"

            with open(backup_file, 'w') as f:
                json.dump({'system_info': system_info, 'results': results}, f, indent=2)
            benchmarker.logger.info(f"Raw results saved as backup to {backup_file}")
        except Exception as backup_error:
            benchmarker.logger.error(f"Failed to save backup results: {backup_error}")
        sys.exit(1)


if __name__ == "__main__":    
    main()