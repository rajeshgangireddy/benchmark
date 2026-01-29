# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Abstract base class for benchmark implementations.
Provides common functionality for timing, logging, and run management.
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any

import torch
from lightning import seed_everything


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmark implementations.
    
    Provides:
    - Logger setup
    - Device synchronization
    - Multi-run execution with seeding
    - Wait time between runs
    """
    
    def __init__(self, args) -> None:
        self.args = args
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _sync_device(self) -> None:
        """Synchronize device to ensure accurate timing."""
        device = getattr(self.args, 'device', 'cpu')
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
        # CPU doesn't need synchronization - operations are synchronous by default

    @abstractmethod
    def create_engine(self) -> Any:
        """Create the training engine. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_datamodule(self) -> Any:
        """Create the data module. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_model(self) -> Any:
        """Create the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def run_single_run(self, engine: Any, model: Any, datamodule: Any) -> tuple[float, float, dict]:
        """
        Execute a single training/testing run.
        
        Returns:
            tuple: (training_time, testing_time, metrics)
        """
        pass

    def measure_time(self, func, *args, **kwargs) -> tuple[float, Any]:
        """
        Measure execution time of a function with device sync.
        
        Returns:
            tuple: (elapsed_time, function_result)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        self._sync_device()
        elapsed = time.perf_counter() - start
        return elapsed, result

    def run_benchmark(self) -> list[dict[str, Any]]:
        """
        Execute multiple benchmark runs with seeding and wait times.
        
        Returns:
            List of result dictionaries from each run.
        """
        num_runs = self.args.num_runs
        self.logger.info(f"Starting benchmark with {num_runs} runs.")
        results = []
        
        for i in range(num_runs):
            run_num = i + 1
            self.logger.info(f"Starting run {run_num}/{num_runs}")
            
            seed = self.args.seed + i
            seed_everything(seed, workers=True, verbose=False)
            self.logger.info(f"Seed: {seed}")
            
            try:
                engine = self.create_engine()
                datamodule = self.create_datamodule()
                model = self.create_model()
                
                training_time, testing_time, metrics = self.run_single_run(engine, model, datamodule)
                
                metrics_dict = metrics[0] if isinstance(metrics, list) and metrics else {}
                if isinstance(metrics, dict):
                    metrics_dict = metrics
                    
                result = {
                    "run_id": run_num,
                    "seed": seed,
                    "training_time_sec": training_time,
                    "testing_time_sec": testing_time,
                    **metrics_dict
                }
                results.append(result)
                self.logger.info(f"Completed run {run_num}/{num_runs}")
                
            except Exception as e:
                self.logger.error(f"Error in run {run_num}: {e}")
                self.logger.error(traceback.format_exc())
            
            if num_runs > 1 and i < num_runs - 1:
                wait_time = self.args.wait_time
                self.logger.info(f"Waiting {wait_time}s before next run")
                time.sleep(wait_time)
        
        return results
