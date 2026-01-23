# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dataset constants and utilities for Anomalib and OTX benchmarking.
"""

MVTEC_CATEGORIES = [
    "carpet",
    "grid",
    "leather", 
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

# OTX supported task types
OTX_TASK_TYPES = [
    "MULTI_CLASS_CLS",
    "MULTI_LABEL_CLS",
    "H_LABEL_CLS",
    "DETECTION",
    "ROTATED_DETECTION",
    "INSTANCE_SEGMENTATION",
    "SEMANTIC_SEGMENTATION",
    "KEYPOINT_DETECTION",
]


def list_otx_models(task: str | None = None, pattern: str | None = None) -> list[str]:
    """
    List available OTX models for a given task.
    
    Args:
        task: OTX task type (e.g., "DETECTION", "MULTI_CLASS_CLS").
        pattern: Optional glob pattern to filter models (not currently supported).
        
    Returns:
        List of model names.
        
    Raises:
        ImportError: If OTX is not installed.
    """
    try:
        from otx.backend.native.cli.utils import list_models
        # Note: pattern parameter is not supported by this OTX version
        return list_models(task=task, print_table=False)
    except ImportError:
        raise ImportError("OTX is not installed. Install with: pip install otx")


def validate_otx_task(task: str) -> bool:
    """Validate that task is a supported OTX task type."""
    return task.upper() in OTX_TASK_TYPES

