# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared statistical functions for benchmark result summarization.
Used by both Anomalib and OTX benchmarking modules.
"""

import statistics
from typing import Any


def summarise_results(
    results: list[dict[str, Any]], 
    time_key: str = "training_time_sec",
    exclude_keys: tuple[str, ...] = ("run_id", "seed")
) -> dict[str, Any]:
    """
    Summarise benchmark results by calculating mean and standard deviation.

    Args:
        results: List of benchmark results from each run.
        time_key: Primary time metric key for summarization.
        exclude_keys: Keys to exclude from metric averaging.

    Returns:
        Summarised results with mean and std for each metric.
    """
    if not results:
        return {}
    
    summarised = {}
    num_runs = len(results)
    
    metric_keys = [key for key in results[0].keys() if key not in exclude_keys]
    
    for key in metric_keys:
        values = [result[key] for result in results if key in result]
        if not values:
            continue
        
        # Skip non-numeric values
        if not isinstance(values[0], (int, float)):
            continue
            
        summarised[f"mean_{key}"] = statistics.mean(values)
        summarised[f"std_{key}"] = statistics.stdev(values) if num_runs > 1 else 0.0

    return summarised


def summarise_results_mlperf(
    results: list[dict[str, Any]], 
    time_key: str = "training_time_sec",
    exclude_keys: tuple[str, ...] = ("run_id", "seed")
) -> dict[str, Any]:
    """
    Summarise results using MLPerf methodology: drop fastest and slowest runs.

    Args:
        results: List of benchmark results from each run.
        time_key: Key used to identify fastest/slowest runs.
        exclude_keys: Keys to exclude from metric averaging.

    Returns:
        Summarised results after filtering outliers.
        
    Raises:
        ValueError: If fewer than 3 results provided.
    """
    if len(results) < 3:
        raise ValueError(f"MLPerf summary requires at least 3 runs, got {len(results)}")

    time_values = [result[time_key] for result in results]
    slowest_idx = time_values.index(max(time_values))
    fastest_idx = time_values.index(min(time_values))

    filtered = [r for i, r in enumerate(results) if i not in (slowest_idx, fastest_idx)]
    return summarise_results(filtered, time_key, exclude_keys)


def summarise_inference_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise inference benchmark results with percentiles.
    
    Args:
        results: List of inference benchmark results from each run.
        
    Returns:
        Dictionary with mean, stdev, and percentiles for timing metrics.
    """
    if not results:
        return {}
    
    summarised = {}
    num_runs = len(results)
    
    # Get keys present in ALL results
    all_keys = set(results[0].keys())
    for result in results[1:]:
        all_keys = all_keys.intersection(result.keys())
    
    metric_keys = [key for key in all_keys if key != "run_id"]
    
    # Priority order for output
    priority_metrics = [
        "fps", "avg_time", "min_time", "max_time",
        "p50_latency", "p95_latency", "p99_latency",
        "total_time", "num_inferences"
    ]
    
    def sort_key(key):
        if key in priority_metrics:
            return (0, priority_metrics.index(key))
        return (1, key)
    
    sorted_keys = sorted(metric_keys, key=sort_key)
    
    for key in sorted_keys:
        values = [result[key] for result in results if key in result]
        if not values or not isinstance(values[0], (int, float)):
            continue
            
        summarised[f"mean_{key}"] = statistics.mean(values)
        summarised[f"std_{key}"] = statistics.stdev(values) if num_runs > 1 else 0.0
        
        # Add percentiles for timing/fps metrics
        if ("time" in key.lower() or "fps" in key.lower()) and num_runs >= 2:
            try:
                quantiles = statistics.quantiles(values, n=100)
                summarised[f"p50_{key}"] = quantiles[49]
                summarised[f"p95_{key}"] = quantiles[94]
                summarised[f"p99_{key}"] = quantiles[98]
            except statistics.StatisticsError:
                pass
    
    # Handle export_time (only in first run)
    if "export_time" in results[0]:
        summarised["export_time"] = results[0]["export_time"]
    
    return summarised


def summarise_inference_results_mlperf(
    results: list[dict[str, Any]], 
    num_runs: int
) -> dict[str, Any]:
    """
    Summarise inference results using MLPerf methodology.
    
    Args:
        results: List of inference benchmark results.
        num_runs: Total number of runs configured.
        
    Returns:
        Summarised results after dropping fastest/slowest.
        
    Raises:
        ValueError: If fewer than 3 runs.
    """
    if num_runs < 3 or len(results) < 3:
        raise ValueError(f"MLPerf summary requires at least 3 runs")

    avg_times = [result["avg_time"] for result in results]
    slowest_idx = avg_times.index(max(avg_times))
    fastest_idx = avg_times.index(min(avg_times))

    filtered = [r for i, r in enumerate(results) if i not in (slowest_idx, fastest_idx)]
    return summarise_inference_results(filtered)


def flatten_system_info(system_info: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested dictionaries for Excel readability.
    
    Converts: {'cpu': {'name': 'Intel', 'cores': 12}}
    To: {'cpu': '--- Section Header ---', 'cpu_name': 'Intel', 'cpu_cores': 12}
    
    Args:
        system_info: Dictionary with potentially nested structures.
        
    Returns:
        Flattened dictionary with section headers.
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
