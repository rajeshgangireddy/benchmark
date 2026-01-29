# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark implementation classes.

Provides base class and framework-specific implementations for training and inference benchmarks.
"""

from src.benchmarkers.base import BaseBenchmark
from src.benchmarkers.otx import OTXBenchmark

# Make anomalib optional since it may not be installed
try:
    from src.benchmarkers.anomalib import AnomalibBenchmark
    __all__ = ["BaseBenchmark", "OTXBenchmark", "AnomalibBenchmark"]
except ImportError:
    AnomalibBenchmark = None
    __all__ = ["BaseBenchmark", "OTXBenchmark"]
