# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utility modules for benchmarking.

Provides configuration management, metrics processing, statistics, and system information.
"""

from src.utils.config import (
    BenchmarkConfig,
    load_config,
    save_config,
    merge_cli_args,
    config_to_args_namespace,
    validate_training_config,
    validate_inference_config,
    get_default_config,
)
from src.utils.metrics import (
    find_latest_otx_workspace,
    load_otx_metrics_csv,
    create_metrics_summary,
    export_detailed_metrics_to_excel,
)
from src.utils.statistics import (
    summarise_results,
    summarise_results_mlperf,
    summarise_inference_results,
    summarise_inference_results_mlperf,
    flatten_system_info,
)
from src.utils.system_info import get_system_info
from src.utils.dataset import OTX_TASK_TYPES, MVTEC_CATEGORIES, list_otx_models

__all__ = [
    # Config
    "BenchmarkConfig",
    "load_config",
    "save_config", 
    "merge_cli_args",
    "config_to_args_namespace",
    "validate_training_config",
    "validate_inference_config",
    "get_default_config",
    # Metrics
    "find_latest_otx_workspace",
    "load_otx_metrics_csv",
    "create_metrics_summary",
    "export_detailed_metrics_to_excel",
    # Statistics
    "summarise_results",
    "summarise_results_mlperf",
    "summarise_inference_results",
    "summarise_inference_results_mlperf",
    "flatten_system_info",
    # System info
    "get_system_info",
    # Dataset
    "OTX_TASK_TYPES",
    "MVTEC_CATEGORIES",
    "list_otx_models",
]
