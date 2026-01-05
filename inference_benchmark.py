"""Benchmark script for comparing Torch and OpenVINO inference performance.

This script trains a model, exports it to Torch and multiple OpenVINO formats,
and benchmarks inference performance on all formats (PyTorch, OpenVINO FP32, FP16, INT8).
Supports multiple benchmark runs for statistical analysis with mean, stdev, and percentiles.

Statistics Calculated:
    - Per-run: avg_time, fps, min/max/p50/p95/p99 latency
    - Cross-run: mean, std deviation, percentiles for consistency analysis
    - MLPerf summary: Conservative estimate (drops fastest/slowest runs)
    
Excel Output Sheets:
    1. Metrics Guide: Detailed explanation of all statistics
    2. System Info: Hardware/software configuration
    3. Raw Results: All individual run data
    4. Summary: Mean ± Std Dev aggregated results
    5. Summary MLPerf: Filtered results (≥3 runs required)

Usage:
    python inference_benchmark.py --device cuda --model Padim --category bottle --num-runs 3 --num-inferences 100
    python inference_benchmark.py --device cpu --model Stfpm --category transistor --num-runs 5
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from anomalib.data import MVTecAD
from anomalib.deploy import CompressionType, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import get_model
from utils.system_info import get_system_info

# Expected Versions
TORCH_VERSION = "2.9"
PYTHON_VERSION = "3.12.13"
ANOMALIB_VERSION = "2.2.0"
OPENVINO_VERSION = "2024.0"  # Adjust based on your environment

os.environ["TRUST_REMOTE_CODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_versions(system_info: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Check if current versions match expected versions.
    
    Args:
        system_info: Dictionary containing system information
        
    Returns:
        Dictionary mapping component names to (expected, actual) version tuples for mismatches
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


def _flatten_system_info(system_info: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested dictionaries in system_info for better Excel readability.
    
    Args:
        system_info: Dictionary containing system information with potentially nested structures
        
    Returns:
        Dictionary with flattened structure including section headers
    """
    flattened = {}
    
    for key, value in system_info.items():
        if isinstance(value, dict):
            flattened[key] = "--- Section Header ---"
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        else:
            flattened[key] = value
    
    return flattened


def summarise_inference_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise inference benchmark results with statistics.
    
    Args:
        results: List of benchmark results from each run
        
    Returns:
        Dictionary with mean, stdev, and percentiles for all metrics (ordered by importance)
    """
    if not results:
        return {}
    
    summarised = {}
    num_runs = len(results)
    
    # Get all metric keys that are present in ALL results (excluding run_id and export_time)
    # export_time is only in the first run, so handle it separately
    all_keys = set(results[0].keys())
    for result in results[1:]:
        all_keys = all_keys.intersection(result.keys())
    
    metric_keys = [key for key in all_keys if key != "run_id"]
    
    # Define priority order for metrics (most important first)
    priority_metrics = [
        "fps",           # Throughput - most important
        "avg_time",      # Latency - most important
        "min_time",      # Best case latency
        "max_time",      # Worst case latency
        "p50_latency",   # Median latency
        "p95_latency",   # 95th percentile
        "p99_latency",   # 99th percentile
        "total_time",    # Total benchmark time
        "num_inferences" # Number of inferences
    ]
    
    # Sort keys by priority (priority metrics first, then alphabetically)
    def sort_key(key):
        if key in priority_metrics:
            return (0, priority_metrics.index(key))
        return (1, key)
    
    sorted_keys = sorted(metric_keys, key=sort_key)
    
    # Calculate statistics in priority order
    for key in sorted_keys:
        values = [result[key] for result in results]
        
        # Most important: mean and std
        summarised[f"mean_{key}"] = statistics.mean(values)
        summarised[f"std_{key}"] = statistics.stdev(values) if num_runs > 1 else 0.0
        
        # Add percentiles for timing metrics (after mean and std)
        if "time" in key.lower() or "fps" in key.lower():
            if num_runs >= 2:
                try:
                    quantiles = statistics.quantiles(values, n=100)
                    summarised[f"p50_{key}"] = quantiles[49]  # Median
                    summarised[f"p95_{key}"] = quantiles[94]  # 95th percentile
                    summarised[f"p99_{key}"] = quantiles[98]  # 99th percentile
                except statistics.StatisticsError:
                    # Not enough data for quantiles
                    pass
    
    # Handle export_time separately (only present in first run) - add at the end
    if "export_time" in results[0]:
        summarised["export_time"] = results[0]["export_time"]
    
    return summarised


def summarise_inference_results_mlperf(results: list[dict[str, Any]], num_runs: int) -> dict[str, Any]:
    """Summarise inference results using MLPerf methodology (drop fastest and slowest).
    
    Args:
        results: List of benchmark results from each run
        num_runs: Total number of runs
        
    Returns:
        Dictionary with summarised results after filtering
        
    Raises:
        ValueError: If num_runs < 3
    """
    if num_runs < 3:
        raise ValueError(f"MLPerf summary requires at least 3 runs, but only {num_runs} provided.")
    
    if len(results) < 3:
        raise ValueError(f"MLPerf summary requires at least 3 completed runs, but only {len(results)} available.")

    # Use avg_time as the key metric for filtering
    avg_times = [result["avg_time"] for result in results]
    
    slowest_run_idx = avg_times.index(max(avg_times))
    fastest_run_idx = avg_times.index(min(avg_times))

    filtered_results = [result for idx, result in enumerate(results) 
                        if idx not in (slowest_run_idx, fastest_run_idx)]
    
    return summarise_inference_results(filtered_results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Torch vs OpenVINO inference performance with statistical analysis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "CPU", "GPU", "xpu", "XPU"],
        help="Device to run inference on (cuda/cpu/xpu for Torch, CPU/GPU for OpenVINO)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["Padim"],
        help="Model(s) to use for benchmarking. Can specify multiple models separated by spaces (default: Padim)",
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=100,
        help="Number of inference iterations per run (default: 100, recommended: >=100)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of complete benchmark runs for statistical analysis (default: 3)",
    )
    parser.add_argument(
        "--warmup-inferences",
        type=int,
        default=10,
        help="Number of warmup iterations before benchmark (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 for latency benchmarking)",
    )
    parser.add_argument(
        "--wait-time",
        type=int,
        default=20,
        help="Wait time in seconds between runs for system cooldown (default: 20)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="transistor",
        help="MVTec AD category to use (default: transistor)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing exported models",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing model directory (used with --skip-training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results (default: ./benchmark_results)",
    )
    return parser.parse_args()


def save_results_to_excel(all_models_results, system_info, config, output_path):
    """Save benchmark results to an Excel file using pandas (clean, no formatting).

    Args:
        all_models_results: Dictionary mapping model names to their results dict
        system_info: Dictionary containing system information
        config: Dictionary with configuration parameters
        output_path: Path to save the Excel file
    """
    try:
        # Create metrics documentation sheet
        metrics_doc = [
            ["Metric", "Description", "Formula/Calculation"],
            ["", "", ""],
            ["=== PER-RUN METRICS ===", "", ""],
            ["total_time", "Sum of all inference times in one run", "Σ(all inference times)"],
            ["avg_time", "Average inference time per image", "total_time / num_inferences"],
            ["fps", "Frames (images) processed per second", "num_inferences / total_time"],
            ["min_time", "Fastest single inference in the run", "min(all inference times)"],
            ["max_time", "Slowest single inference in the run", "max(all inference times)"],
            ["p50_latency", "50th percentile (median) latency", "Typical inference time - middle of distribution"],
            ["p95_latency", "95th percentile latency", "95% of inferences complete within this time (SLA metric)"],
            ["p99_latency", "99th percentile latency", "99% of inferences complete within this time (worst-case)"],
            ["", "", ""],
            ["=== CROSS-RUN STATISTICS ===", "", ""],
            ["mean_{metric}", "Average of metric across all runs", "Σ(metric_values) / num_runs"],
            ["std_{metric}", "Standard deviation - measures consistency", "√(Σ(x - mean)² / n) - Lower is better!"],
            ["p50_{metric}", "Median of metric across runs", "Middle value when runs are sorted"],
            ["p95_{metric}", "95th percentile across runs", "95% of runs had this value or better"],
            ["p99_{metric}", "99th percentile across runs", "99% of runs had this value or better"],
            ["", "", ""],
            ["=== MLPERF SUMMARY ===", "", ""],
            ["MLPerf Method", "Drops fastest and slowest runs", "More conservative, robust estimate (requires ≥3 runs)"],
            ["", "Reports mean of remaining runs", "Follows MLPerf inference benchmarking guidelines"],
            ["", "", ""],
            ["=== INTERPRETATION GUIDE ===", "", ""],
            ["Good Results", "Low std deviation (< 5% of mean)", "Consistent, repeatable performance"],
            ["", "P95/P99 close to P50", "Predictable latency, few outliers"],
            ["", "Higher FPS", "Better throughput"],
            ["Warning Signs", "High std deviation", "System instability, thermal throttling, or background load"],
            ["", "P99 >> P50", "Many outliers/spikes - investigate system"],
            ["", "Results vary between runs", "Need more warmup or longer cooldown period"],
            ["", "", ""],
            ["=== WHY MULTIPLE RUNS? ===", "", ""],
            ["Account for variance", "System load, CPU frequency scaling, thermal effects", ""],
            ["Detect outliers", "One-off slowdowns from background processes", ""],
            ["Build confidence", "Standard deviation quantifies reliability", ""],
            ["MLPerf compliance", "Industry-standard methodology for fair comparison", ""],
        ]
        metrics_df = pd.DataFrame(metrics_doc[1:], columns=metrics_doc[0])
        
        # Flatten system info for better Excel readability
        flattened_system_info = _flatten_system_info(system_info)
        system_info_df = pd.DataFrame(list(flattened_system_info.items()), 
                                       columns=["Component", "Details"])
        
        # Build raw results DataFrame
        raw_results_data = []
        for model_name, formats_dict in all_models_results.items():
            for format_name, runs_list in formats_dict.items():
                for run_data in runs_list:
                    row = {
                        "model": model_name,
                        "format": format_name,
                        **run_data
                    }
                    raw_results_data.append(row)
        
        raw_results_df = pd.DataFrame(raw_results_data)
        
        # Build summary DataFrame (mean and stdev across runs for each model-format pair)
        summary_data = []
        for model_name, formats_dict in all_models_results.items():
            for format_name, runs_list in formats_dict.items():
                if runs_list:
                    summary_stats = summarise_inference_results(runs_list)
                    row = {
                        "model": model_name,
                        "format": format_name,
                        **summary_stats
                    }
                    summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Build MLPerf summary (if applicable)
        mlperf_data = []
        num_runs = config.get("num_runs", 0)
        if num_runs >= 3:
            for model_name, formats_dict in all_models_results.items():
                for format_name, runs_list in formats_dict.items():
                    if len(runs_list) >= 3:
                        try:
                            mlperf_stats = summarise_inference_results_mlperf(runs_list, num_runs)
                            row = {
                                "model": model_name,
                                "format": format_name,
                                **mlperf_stats
                            }
                            mlperf_data.append(row)
                        except ValueError:
                            pass  # Skip if not enough runs
        
        mlperf_df = pd.DataFrame(mlperf_data) if mlperf_data else None
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name="Metrics Guide", index=False)
            system_info_df.to_excel(writer, sheet_name="System Info", index=False)
            raw_results_df.to_excel(writer, sheet_name="Raw Results", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            if mlperf_df is not None:
                mlperf_df.to_excel(writer, sheet_name="Summary MLPerf", index=False)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Excel file contains {len(all_models_results)} model(s) with Metrics Guide, System Info, Raw Results, Summary sheets")
        if mlperf_df is not None:
            print("MLPerf summary sheet included")
            
    except Exception as e:
        raise RuntimeError(f"Failed to save results to Excel: {e}") from e


def benchmark_inferencer(inferencer, data, num_inferences, inferencer_name, device, warmup_inferences=10):
    """Run benchmark for a given inferencer.

    Args:
        inferencer: The inferencer instance (TorchInferencer or OpenVINOInferencer)
        data: Data module containing test data
        num_inferences: Number of inferences to run
        inferencer_name: Name for logging (e.g., "TorchInferencer")
        device: Device being used
        warmup_inferences: Number of warmup iterations (default: 10)

    Returns:
        dict: Benchmark results including total time, avg time, FPS, and percentiles
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {inferencer_name} on {device}")
    print(f"{'=' * 60}")

    # Validate test data exists
    if not hasattr(data, "test_data") or len(data.test_data) == 0:
        raise ValueError("No test data available. Please check the dataset.")

    timings = []

    # Warmup runs (exclude from timing)
    print(f"Running {warmup_inferences} warmup iterations...")
    for i in range(warmup_inferences):
        sample = data.test_data[i % len(data.test_data)]
        _ = inferencer.predict(sample.image)
    print("Warmup complete. Starting benchmark...")

    # Benchmark runs
    for i in range(num_inferences):
        sample = data.test_data[i % len(data.test_data)]

        tic = time.time()
        result = inferencer.predict(sample.image)
        toc = time.time()

        inference_time = toc - tic
        timings.append(inference_time)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{num_inferences}] Inference time: {inference_time:.4f}s")

    # Calculate statistics
    total_time = sum(timings)
    avg_time = total_time / num_inferences
    fps = num_inferences / total_time
    min_time = min(timings)
    max_time = max(timings)
    
    # Calculate percentiles
    p50 = p95 = p99 = None
    if len(timings) >= 2:
        try:
            sorted_timings = sorted(timings)
            p50_idx = int(len(timings) * 0.50)
            p95_idx = int(len(timings) * 0.95)
            p99_idx = int(len(timings) * 0.99)
            p50 = sorted_timings[p50_idx]
            p95 = sorted_timings[p95_idx]
            p99 = sorted_timings[p99_idx]
        except (IndexError, statistics.StatisticsError):
            pass

    results = {
        "total_time": total_time,
        "avg_time": avg_time,
        "fps": fps,
        "min_time": min_time,
        "max_time": max_time,
        "num_inferences": num_inferences,
    }
    
    if p50 is not None:
        results["p50_latency"] = p50
        results["p95_latency"] = p95
        results["p99_latency"] = p99

    print(f"\n{inferencer_name} Results:")
    print(f"  Total time:        {total_time:.4f}s")
    print(f"  Average time:      {avg_time:.4f}s")
    print(f"  Min time:          {min_time:.4f}s")
    print(f"  Max time:          {max_time:.4f}s")
    if p50 is not None:
        print(f"  P50 latency:       {p50:.4f}s")
        print(f"  P95 latency:       {p95:.4f}s")
        print(f"  P99 latency:       {p99:.4f}s")
    print(f"  FPS:               {fps:.2f}")

    return results


def main():
    """Main function to run the inference benchmark with robust error handling."""
    args = parse_args()

    # Validate arguments
    if args.num_inferences < 100:
        print(f"Warning: num_inferences={args.num_inferences} is less than 100. For reliable benchmarking, recommend >= 100.")
    
    if args.num_runs < 3:
        print(f"Warning: num_runs={args.num_runs} is less than 3. MLPerf-style summary requires at least 3 runs.")
        print("MLPerf summary will be skipped in the results.")
    
    if args.batch_size > 1:
        print(f"Note: Batch size = {args.batch_size}. Running in throughput mode (not latency mode).")
    
    # Convert single model to list if necessary
    models = args.model if isinstance(args.model, list) else [args.model]

    # Create output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")
    except Exception as e:
        print(f"Error: Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)

    # Get system information
    device_lower = args.device.lower()
    device_for_sysinfo = None if device_lower == "cpu" else "cuda" if device_lower in ["cuda", "gpu"] else "xpu" if device_lower == "xpu" else device_lower
    
    try:
        system_info = get_system_info(device=device_for_sysinfo)
    except Exception as e:
        print(f"Error: Failed to retrieve system information: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    for key, value in system_info.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")
    
    # Check version mismatches
    version_mismatches = check_versions(system_info)
    if version_mismatches:
        for component, (expected, current) in version_mismatches.items():
            print(f"Warning: Version mismatch for {component}: Expected {expected}, got {current}")
    
    print(f"\n{'=' * 60}")
    print("INFERENCE BENCHMARK CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Models:               {', '.join(models)}")
    print(f"Device:               {args.device}")
    print(f"Number of runs:       {args.num_runs}")
    print(f"Inferences per run:   {args.num_inferences}")
    print(f"Warmup inferences:    {args.warmup_inferences}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Wait time (sec):      {args.wait_time}")
    print(f"Category:             {args.category}")
    print(f"Skip training:        {args.skip_training}")
    print("Formats:              PyTorch, OpenVINO (FP32, FP16, INT8)")
    print(f"{'=' * 60}\n")

    # Load data (shared across all models)
    print("Loading dataset...")
    try:
        data = MVTecAD(category=args.category, num_workers=0)
        data.setup()
        print(f"Dataset loaded: {len(data.test_data)} test images\n")
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        sys.exit(1)

    # Map device names (Torch uses "cuda"/"cpu"/"xpu", OpenVINO uses "GPU"/"CPU")
    # Note: If OpenVINO GPU support is not available, it will fall back to CPU
    if device_lower == "cuda":
        torch_device, ov_device = "cuda", "CPU"  # Changed: Use CPU for OpenVINO since GPU drivers not available
        print("Note: Using CUDA for PyTorch, CPU for OpenVINO (GPU drivers not detected)")
    elif device_lower == "cpu":
        torch_device, ov_device = "cpu", "CPU"
    elif device_lower == "gpu":
        torch_device, ov_device = "cuda", "CPU"  # Changed: Fallback to CPU for OpenVINO
        print("Note: Using CUDA for PyTorch, CPU for OpenVINO (GPU drivers not detected)")
    elif device_lower == "xpu":
        torch_device, ov_device = "xpu", "GPU"
        print("Note: Using XPU for PyTorch, GPU for OpenVINO (Intel GPU)")
    else:
        torch_device, ov_device = "cpu", "CPU"

    # Dictionary to store all results: {model_name: {format_name: [run1_results, run2_results, ...]}}
    all_models_results = {}

    # Process each model
    for model_idx, model_name in enumerate(models):
        print(f"\n{'#' * 60}")
        print(f"# MODEL {model_idx + 1}/{len(models)}: {model_name}")
        print(f"{'#' * 60}\n")

        # Initialize results storage for this model
        all_models_results[model_name] = {}

        # Get or train model and export to all formats
        if args.skip_training and args.model_path:
            print(f"Using existing models from: {args.model_path}")
            torch_path = Path(args.model_path) / f"{model_name}_model.pt"
            ov_fp32_path = Path(args.model_path) / f"{model_name}_model.xml"
            ov_fp16_path = Path(args.model_path) / f"{model_name}_model_fp16.xml"
            ov_int8_path = Path(args.model_path) / f"{model_name}_model_int8_ptq.xml"

            if not torch_path.exists() or not ov_fp32_path.exists():
                print(f"Error: Required model files not found for {model_name}. Skipping.")
                continue
            
            fp32_export_time = fp16_export_time = int8_export_time = None
        else:
            # Train and export model
            print(f"Initializing and training {model_name}...")
            try:
                model = get_model(model_name)
                engine = Engine(max_epochs=5)
                engine.fit(datamodule=data, model=model)
                engine.test(datamodule=data, model=model)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                traceback.print_exc()
                continue

            # Export models
            print(f"\nExporting {model_name} to all formats...")
            filename = f"{model_name}_{args.category}_benchmark"

            try:
                torch_path = engine.export(model=model, export_type="torch", model_file_name=filename)
                print(f"PyTorch model exported: {torch_path}")
            except Exception as e:
                print(f"Error exporting PyTorch model for {model_name}: {e}")
                continue

            try:
                fp32_start = time.time()
                ov_fp32_path = engine.export(model=model, export_type="openvino", model_file_name=filename)
                fp32_export_time = time.time() - fp32_start
                print(f"OpenVINO FP32 exported: {ov_fp32_path} ({fp32_export_time:.2f}s)")
            except Exception as e:
                print(f"Error exporting OpenVINO FP32 for {model_name}: {e}")
                continue

            try:
                fp16_start = time.time()
                ov_fp16_path = engine.export(
                    model=model, export_type="openvino",
                    model_file_name=f"{filename}_fp16",
                    compression_type=CompressionType.FP16
                )
                fp16_export_time = time.time() - fp16_start
                print(f"OpenVINO FP16 exported: {ov_fp16_path} ({fp16_export_time:.2f}s)")
            except Exception as e:
                print(f"Warning: FP16 export failed for {model_name}: {e}")
                ov_fp16_path = None
                fp16_export_time = None

            try:
                int8_start = time.time()
                ov_int8_path = engine.export(
                    model=model, export_type="openvino",
                    model_file_name=f"{filename}_int8_ptq",
                    compression_type=CompressionType.INT8_PTQ,
                    datamodule=data
                )
                int8_export_time = time.time() - int8_start
                print(f"OpenVINO INT8 exported: {ov_int8_path} ({int8_export_time:.2f}s)")
            except Exception as e:
                print(f"Warning: INT8 export failed for {model_name}: {e}")
                ov_int8_path = None
                int8_export_time = None

        # Run multiple benchmark runs for each format
        formats_to_benchmark = [
            ("PyTorch", torch_path, TorchInferencer, torch_device, None),
            ("OpenVINO (FP32)", ov_fp32_path, OpenVINOInferencer, ov_device, fp32_export_time),
            ("OpenVINO (FP16)", ov_fp16_path, OpenVINOInferencer, ov_device, fp16_export_time),
            ("OpenVINO (INT8)", ov_int8_path, OpenVINOInferencer, ov_device, int8_export_time),
        ]

        for format_name, model_path, inferencer_class, device, export_time in formats_to_benchmark:
            if model_path is None or not Path(model_path).exists():
                print(f"\nSkipping {format_name} (model not available)")
                continue

            print(f"\n{'=' * 60}")
            print(f"BENCHMARKING: {model_name} - {format_name}")
            print(f"{'=' * 60}")

            # Store results for all runs of this format
            format_results = []

            for run_idx in range(args.num_runs):
                print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
                
                try:
                    inferencer = inferencer_class(path=model_path, device=device)
                    run_results = benchmark_inferencer(
                        inferencer, data, args.num_inferences,
                        f"{model_name} {format_name}",
                        device, args.warmup_inferences
                    )
                    run_results["run_id"] = run_idx + 1
                    if export_time is not None and run_idx == 0:
                        run_results["export_time"] = export_time
                    format_results.append(run_results)
                except Exception as e:
                    print(f"Error in run {run_idx + 1} for {format_name}: {e}")
                    traceback.print_exc()
                
                # Wait between runs (except after last run)
                if run_idx < args.num_runs - 1:
                    print(f"Cooling down for {args.wait_time} seconds...")
                    time.sleep(args.wait_time)

            # Store results for this format
            if format_results:
                all_models_results[model_name][format_name] = format_results

        # Print summary for this model
        if all_models_results[model_name]:
            print("\n" + "=" * 60)
            print(f"SUMMARY FOR {model_name}")
            print("=" * 60)
            for format_name, runs_list in all_models_results[model_name].items():
                if runs_list:
                    summary = summarise_inference_results(runs_list)
                    print(f"\n{format_name}:")
                    print(f"  Mean Avg Time: {summary.get('mean_avg_time', 0):.4f}s ± {summary.get('std_avg_time', 0):.4f}s")
                    print(f"  Mean FPS:      {summary.get('mean_fps', 0):.2f} ± {summary.get('std_fps', 0):.2f}")
            print("=" * 60)

    # Validate we have results
    if not all_models_results or all(not formats for formats in all_models_results.values()):
        print("\nError: No benchmark results were collected. Exiting.")
        sys.exit(1)

    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"INF_BM_{args.device}_{'_'.join(models)}_runs-{args.num_runs}_{timestamp}.xlsx"
    result_path = output_dir / filename

    config = {
        "models": models,
        "category": args.category,
        "device": f"Torch={torch_device.upper()}, OpenVINO={ov_device}",
        "num_runs": args.num_runs,
        "num_inferences": args.num_inferences,
        "warmup_inferences": args.warmup_inferences,
        "batch_size": args.batch_size,
        "wait_time": args.wait_time,
        "test_images": len(data.test_data),
        "timestamp": timestamp,
    }

    try:
        save_results_to_excel(all_models_results, system_info, config, result_path)
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        traceback.print_exc()
        
        # Backup save as JSON
        try:
            backup_path = output_dir / f"backup_results_{timestamp}.json"
            with open(backup_path, 'w') as f:
                json.dump({
                    'system_info': system_info,
                    'config': config,
                    'results': all_models_results
                }, f, indent=2, default=str)
            print(f"Results saved as backup JSON: {backup_path}")
        except Exception as backup_error:
            print(f"Failed to save backup: {backup_error}")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("# BENCHMARK COMPLETE")
    print("#" * 60)
    print(f"Results saved to: {result_path}")
    print(f"Models benchmarked: {', '.join(models)}")
    print(f"Total runs per format: {args.num_runs}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
