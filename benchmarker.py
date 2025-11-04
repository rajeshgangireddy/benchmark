# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmarking Tool for Anomalib Models.
"""


import argparse
import logging
import time
import traceback
from typing import Any

import torch
from lightning import seed_everything

from anomalib.data import AnomalibDataModule, MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import get_model, AnomalibModule
from anomalib.callbacks import ModelCheckpoint

from utils.dataset import MVTEC_CATEGORIES

# Constants
MS_TO_SEC_CONVERSION = 1000.0


class AnomalibBenchmark():
    """
    Benchmarking class for Anomalib models.
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Sets up the logger for benchmarking.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("benchmark.log"),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)
    

    def create_engine(self) -> Engine:
        """
        Creates the Anomalib engine based on the provided arguments.

        Returns:
            Engine: Configured engine instance.
        """

        max_epochs = self.args.max_epochs
        device = self.args.device
        save_checkpoint = self.args.save_checkpoint_during_training

        if save_checkpoint:
            self.logger.warning("Saving checkpoints during training is enabled. "
                                "This may slow down the training process depending on disk speeds.")
            model_checkpoint_cv = None # Uses anomalib's default checkpoint callback which saves checkpoints every epoch
        else:
            model_checkpoint_cv = ModelCheckpoint(save_top_k=0)  # Disable checkpoint saving
        
        callbacks = [model_checkpoint_cv] if model_checkpoint_cv is not None else []

        if device == "xpu":
            self.logger.info("Creating Engine with XPU Strategy and Accelerator.")
            return Engine(
                strategy=SingleXPUStrategy(),
                accelerator=XPUAccelerator(),
                max_epochs=max_epochs,
                callbacks=callbacks
            )
        elif device == "cpu":
            self.logger.info("Creating Engine with CPU Accelerator.")
            return Engine(
            accelerator="cpu",
            max_epochs=max_epochs,
            callbacks=callbacks
            )
        elif device == "cuda":
            # For cuda simply return default engine
            self.logger.info("Creating Engine with default (CUDA)settings.")
            return Engine(max_epochs=max_epochs, callbacks=callbacks)
        else:
            raise ValueError(f"Unsupported device type: {device}. Supported devices are 'cpu', 'cuda', and 'xpu'.")

    def create_datamodule(self) -> AnomalibDataModule:
        """
        Creates the Anomalib data module based on the provided arguments.

        Returns:
            AnomalibDataModule: Configured data module instance.
        """
        train_batchsize = self.args.train_batch_size
        eval_batch_size = self.args.eval_batch_size
        num_workers = self.args.num_workers
        category = self.args.category


        if category not in MVTEC_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {MVTEC_CATEGORIES}.")

        self.logger.info(f"Creating MVTecAD DataModule for category: {category} "
                         f"with train batch size: {train_batchsize}, "
                         f"eval batch size: {eval_batch_size}, "
                         f"num_workers: {num_workers}.")
        return MVTecAD(category=category,
                       train_batch_size=train_batchsize,
                       eval_batch_size=eval_batch_size,
                       num_workers=num_workers
                       )
    def create_anomalib_model(self) -> AnomalibModule:
        """
        Creates the Anomalib model based on the provided arguments.

        Returns:
            Any: Configured Anomalib model instance.
        """
        model_name = self.args.model_name
        self.logger.info(f"Creating Anomalib model: {model_name}.")
        model = get_model(model=model_name)
        return model
    def _sync_torch(self) -> None:
        """
        Synchronizes the PyTorch process based on the device type.
        """
        device = self.args.device
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "xpu" and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif device == "cpu":
            pass  # CPU operations are synchronous by default
        else:
            self.logger.debug(f"No explicit synchronization needed for device '{device}'.")
          

    def run_single_run(
        self, 
        engine: Engine, 
        model: AnomalibModule, 
        datamodule: AnomalibDataModule
    ) -> tuple[float, float, dict]:
        """
        Executes a single training and testing run.
        
        Args:
            engine: Anomalib Engine instance
            model: Anomalib model instance
            datamodule: Anomalib data module instance
            
        Returns:
            tuple: (training_time, testing_time, metrics)
        """
        device = self.args.device
        
        training_time = self._measure_training_time(engine, model, datamodule, device)
        self.logger.info(f"Training completed in {training_time:.2f} seconds.")

        testing_time, metrics = self._measure_testing_time(engine, model, datamodule, device)
        self.logger.info(f"Testing completed in {testing_time:.2f} seconds.")
        
        return training_time, testing_time, metrics
    
    def _measure_training_time(
        self, 
        engine: Engine, 
        model: AnomalibModule, 
        datamodule: AnomalibDataModule, 
        device: str
    ) -> float:
        """Measure training time using device-specific timing mechanism."""
        if device == "cuda":
            train_start = torch.cuda.Event(enable_timing=True)
            train_end = torch.cuda.Event(enable_timing=True)
            train_start.record()
            engine.fit(model=model, datamodule=datamodule)
            train_end.record()
            self._sync_torch()
            return train_start.elapsed_time(train_end) / MS_TO_SEC_CONVERSION
        elif device == "xpu":
            train_start = torch.xpu.Event(enable_timing=True)
            train_end = torch.xpu.Event(enable_timing=True)
            train_start.record()
            engine.fit(model=model, datamodule=datamodule)
            train_end.record()
            self._sync_torch()
            return train_start.elapsed_time(train_end) / MS_TO_SEC_CONVERSION
        else:  # CPU or fallback
            train_start_time = time.perf_counter()
            engine.fit(model=model, datamodule=datamodule)
            return time.perf_counter() - train_start_time
    
    def _measure_testing_time(
        self, 
        engine: Engine, 
        model: AnomalibModule, 
        datamodule: AnomalibDataModule, 
        device: str
    ) -> tuple[float, dict]:
        """Measure testing time using device-specific timing mechanism."""
        if device == "cuda":
            test_start = torch.cuda.Event(enable_timing=True)
            test_end = torch.cuda.Event(enable_timing=True)
            test_start.record()
            metrics = engine.test(model=model, datamodule=datamodule)
            test_end.record()
            self._sync_torch()
            return test_start.elapsed_time(test_end) / MS_TO_SEC_CONVERSION, metrics
        elif device == "xpu":
            test_start = torch.xpu.Event(enable_timing=True)
            test_end = torch.xpu.Event(enable_timing=True)
            test_start.record()
            metrics = engine.test(model=model, datamodule=datamodule)
            test_end.record()
            self._sync_torch()
            return test_start.elapsed_time(test_end) / MS_TO_SEC_CONVERSION, metrics
        else:  # CPU or fallback
            test_start_time = time.perf_counter()
            metrics = engine.test(model=model, datamodule=datamodule)
            return time.perf_counter() - test_start_time, metrics
    
        
    def run_benchmark(self) -> list[dict[str, Any]]:
        """
        Runs the benchmark for the specified number of runs.
        Returns:
            list[dict[str, Any]]: Benchmark results with training and testing times along with metrics.
        """
        num_runs = self.args.num_runs
        self.logger.info(f"Starting benchmark with {num_runs} runs.")
        results = []
        for i in range(num_runs):
            run_num = i+1
            self.logger.info(f"Starting benchmark run {run_num}/{num_runs}.")
            seed = self.args.seed + i
            seed_everything(seed, workers=True, verbose=True)
            self.logger.info(f"Using seed: {seed} for run {run_num}.")
            try:
                engine = self.create_engine()
                datamodule = self.create_datamodule()
                model = self.create_anomalib_model()
                training_time, testing_time, metrics = self.run_single_run(engine, model, datamodule)
                # Extract metrics from the list returned by engine.test()
                metrics_dict = metrics[0] if isinstance(metrics, list) and len(metrics) > 0 else {}
                result = {
                    "run_id": run_num,
                    "seed": seed,
                    "training_time_sec": training_time,
                    "testing_time_sec": testing_time,
                    **metrics_dict
                }
                results.append(result)
                self.logger.info(f"Completed benchmark run {run_num}/{num_runs}.")
            except Exception as e:
                self.logger.error(f"Error during benchmark run {run_num}/{num_runs}: {e}")
                self.logger.error(traceback.format_exc())
            if num_runs > 1 and i < num_runs - 1:
                wait_time = self.args.wait_time
                self.logger.info(f"Waiting for {wait_time} seconds before next run.")
                time.sleep(wait_time)
        # Raw Results 
        return results
    

                

                





