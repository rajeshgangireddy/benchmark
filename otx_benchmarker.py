# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmarking implementation for OTX (OpenVINO Training Extensions) models.
"""

import argparse
from pathlib import Path
from typing import Any

from benchmarker_base import BaseBenchmark
from utils.dataset import OTX_TASK_TYPES


class OTXBenchmark(BaseBenchmark):
    """
    Benchmarking class for OTX models.
    
    Supports training benchmarks across various tasks:
    - Classification (multi-class, multi-label, hierarchical)
    - Detection (standard, rotated, keypoint)
    - Segmentation (instance, semantic)
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._validate_args()
        
    def _validate_args(self) -> None:
        """Validate OTX-specific arguments."""
        task = self.args.task.upper()
        if task not in OTX_TASK_TYPES:
            raise ValueError(f"Invalid task '{task}'. Must be one of {OTX_TASK_TYPES}")

    def create_engine(self) -> Any:
        """Create OTX engine based on device and configuration."""
        from otx.engine import Engine as OTXEngine
        
        device = self.args.device
        work_dir = Path(self.args.output_dir) / "otx_workspace"
        
        # Map device names to OTX DeviceType
        device_map = {
            "cpu": "cpu",
            "cuda": "gpu",
            "xpu": "xpu",
        }
        otx_device = device_map.get(device, "auto")
        
        self.logger.info(f"Creating OTX Engine with device={otx_device}")
        
        # Build engine kwargs
        engine_kwargs = {
            "work_dir": str(work_dir),
            "device": otx_device,
        }
        
        # Use model name or recipe path
        model_spec = self.args.model
        if model_spec.endswith(('.yaml', '.yml')):
            # Recipe path provided
            self.logger.info(f"Using recipe: {model_spec}")
            engine_kwargs["model"] = model_spec
        else:
            # Model name provided - will be handled by AutoConfigurator in create_model
            pass
        
        # Data configuration
        engine_kwargs["data"] = self._get_data_root()
        
        return OTXEngine(**engine_kwargs)

    def _get_data_root(self) -> str:
        """Get data root path from args."""
        return self.args.data_root

    def create_datamodule(self) -> Any:
        """
        Create OTX data module.
        
        Note: OTX Engine handles data module creation internally when 
        data_root is provided, but we can also create it explicitly.
        """
        from otx.data import OTXDataModule
        from otx.config.data import SubsetConfig
        from otx.types.task import OTXTaskType
        
        task = OTXTaskType(self.args.task.upper())
        
        train_config = SubsetConfig(
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
        )
        val_config = SubsetConfig(
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
        )
        
        self.logger.info(f"Creating OTX DataModule: task={task}, "
                         f"train_batch={self.args.train_batch_size}, "
                         f"val_batch={self.args.eval_batch_size}")
        
        return OTXDataModule(
            task=task,
            data_root=self.args.data_root,
            train_subset=train_config,
            val_subset=val_config,
        )

    def create_model(self) -> Any:
        """
        Create OTX model.
        
        Uses AutoConfigurator when model name is provided,
        or loads from recipe when path is provided.
        """
        model_spec = self.args.model
        
        if model_spec.endswith(('.yaml', '.yml')):
            # Model will be created by engine from recipe
            self.logger.info(f"Model will be loaded from recipe: {model_spec}")
            return None
        
        # Use AutoConfigurator for model name
        from otx.tools.auto_configurator import AutoConfigurator
        from otx.types.task import OTXTaskType
        
        task = OTXTaskType(self.args.task.upper())
        
        self.logger.info(f"Creating model '{model_spec}' for task '{task}' via AutoConfigurator")
        
        auto_config = AutoConfigurator(
            data_root=self.args.data_root,
            task=task,
            model=model_spec,
        )
        
        datamodule = auto_config.get_datamodule()
        model = auto_config.get_model(label_info=datamodule.label_info)
        
        return model

    def run_single_run(
        self, 
        engine: Any, 
        model: Any, 
        datamodule: Any
    ) -> tuple[float, float, dict]:
        """Execute a single OTX training run."""
        
        # Build training kwargs
        train_kwargs = {
            "max_epochs": self.args.max_epochs,
            "deterministic": False,
        }
        
        if self.args.precision:
            train_kwargs["precision"] = self.args.precision
            self.logger.info(f"Using precision: {self.args.precision}")
        
        # OTX Engine handles model internally if loaded from recipe
        if model is not None:
            # Update engine with explicit model
            engine._model = model
        
        training_time, train_result = self.measure_time(engine.train, **train_kwargs)
        self.logger.info(f"Training completed in {training_time:.2f}s")
        
        # OTX may not always have a separate test phase
        # Get metrics from training result or run test if needed
        metrics = train_result if isinstance(train_result, dict) else {}
        
        # Attempt to run test phase if datamodule has test data
        testing_time = 0.0
        try:
            if hasattr(engine, 'test'):
                testing_time, test_result = self.measure_time(engine.test)
                self.logger.info(f"Testing completed in {testing_time:.2f}s")
                if isinstance(test_result, dict):
                    metrics.update(test_result)
        except Exception as e:
            self.logger.warning(f"Test phase skipped or failed: {e}")
        
        return training_time, testing_time, metrics


class OTXInferenceBenchmark:
    """
    Wrapper for OTX's built-in inference benchmarking.
    
    Uses engine.benchmark() method for latency/throughput measurement,
    wrapped with multi-run averaging for statistical robustness.
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        
    def run_benchmark(self) -> list[dict[str, Any]]:
        """
        Run multiple inference benchmark iterations.
        
        Returns:
            List of result dictionaries with latency, throughput, complexity.
        """
        from otx.engine import Engine as OTXEngine
        import time
        
        results = []
        
        for run_idx in range(self.args.num_runs):
            print(f"Inference benchmark run {run_idx + 1}/{self.args.num_runs}")
            
            # Create engine from checkpoint
            engine = OTXEngine(
                model=self.args.checkpoint,
                data=self.args.data_root,
                device=self.args.device,
            )
            
            # Run OTX's built-in benchmark
            benchmark_result = engine.benchmark(
                batch_size=self.args.batch_size,
                n_iters=self.args.num_inferences,
                extended_stats=True,
                print_table=False,
            )
            
            # Parse results (OTX returns strings like "0.123 s", "45.6 FPS")
            result = {
                "run_id": run_idx + 1,
                "latency_sec": float(benchmark_result.get("latency", "0").replace(" s", "")),
                "throughput_fps": float(benchmark_result.get("throughput", "0").replace(" FPS", "")),
            }
            
            if "complexity" in benchmark_result:
                result["complexity_macs"] = benchmark_result["complexity"]
            
            results.append(result)
            
            if run_idx < self.args.num_runs - 1:
                print(f"Waiting {self.args.wait_time}s before next run...")
                time.sleep(self.args.wait_time)
        
        return results
