# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmarking implementation for Anomalib models.
"""

import argparse
import time
from typing import Any

from anomalib.data import AnomalibDataModule, MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import get_model, AnomalibModule
from anomalib.callbacks import ModelCheckpoint

from src.benchmarkers.base import BaseBenchmark
from src.utils.dataset import MVTEC_CATEGORIES


class AnomalibBenchmark(BaseBenchmark):
    """Benchmarking class for Anomalib models."""
    
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

    def create_engine(self) -> Engine:
        """Create Anomalib engine based on device and barebones settings."""
        max_epochs = self.args.max_epochs
        device = self.args.device
        barebones = self.args.barebones
        precision = getattr(self.args, 'precision', None)

        if barebones:
            self.logger.info("Barebones mode: disabling logging, progress bars, checkpointing")
            common_params = {
                "max_epochs": max_epochs,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "callbacks": [ModelCheckpoint(save_top_k=0)],
            }
        else:
            self.logger.info("Standard mode: logging and checkpointing enabled")
            common_params = {"max_epochs": max_epochs}
        
        if precision:
            common_params["precision"] = precision
            self.logger.info(f"Using precision: {precision}")
                
        if device == "xpu":
            self.logger.info("Creating Engine with XPU Strategy and Accelerator")
            return Engine(
                strategy=SingleXPUStrategy(),
                accelerator=XPUAccelerator(),
                **common_params
            )
        elif device == "cpu":
            self.logger.info("Creating Engine with CPU Accelerator")
            return Engine(accelerator="cpu", **common_params)
        elif device == "cuda":
            self.logger.info("Creating Engine with CUDA settings")
            return Engine(**common_params)
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cpu', 'cuda', or 'xpu'.")

    def create_datamodule(self) -> AnomalibDataModule:
        """Create MVTecAD data module."""
        category = self.args.category

        if category not in MVTEC_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {MVTEC_CATEGORIES}.")

        self.logger.info(f"Creating MVTecAD DataModule: category={category}, "
                         f"train_batch={self.args.train_batch_size}, "
                         f"eval_batch={self.args.eval_batch_size}")
        return MVTecAD(
            category=category,
            train_batch_size=self.args.train_batch_size,
            eval_batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers
        )

    def create_model(self) -> AnomalibModule:
        """Create Anomalib model by name."""
        model_name = self.args.model_name
        self.logger.info(f"Creating model: {model_name}")
        return get_model(model=model_name)

    def run_single_run(
        self, 
        engine: Engine, 
        model: AnomalibModule, 
        datamodule: AnomalibDataModule
    ) -> tuple[float, float, dict]:
        """Execute a single training and testing run."""
        training_time, _ = self.measure_time(engine.fit, model=model, datamodule=datamodule)
        self.logger.info(f"Training completed in {training_time:.2f}s")

        testing_time, metrics = self.measure_time(engine.test, model=model, datamodule=datamodule)
        self.logger.info(f"Testing completed in {testing_time:.2f}s")
        
        return training_time, testing_time, metrics
