# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmarking implementation for OTX (OpenVINO Training Extensions) models.

Uses OTX recipe files directly - all training configuration comes from the recipe.
Runs OTX via subprocess to ensure full compatibility with OTX CLI.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.benchmarkers.base import BaseBenchmark


class OTXBenchmark(BaseBenchmark):
    """
    Benchmarking class for OTX models using recipe files.
    
    The recipe file provides all training configuration (model, epochs, 
    batch_size, callbacks, etc.). This class handles:
    - Multi-run execution for statistics
    - Timing measurement via subprocess calls to `otx train`
    - Results collection
    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._validate_args()
        
    def _validate_args(self) -> None:
        """Validate required arguments."""
        if not hasattr(self.args, 'recipe') or not self.args.recipe:
            raise ValueError("Recipe file path is required (--recipe)")
        
        recipe_path = Path(self.args.recipe)
        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
        
        if not hasattr(self.args, 'data_root') or not self.args.data_root:
            raise ValueError("Data root path is required (--data_root)")

    def create_engine(self) -> Any:
        """
        For subprocess-based approach, we don't create an engine object.
        Returns the command arguments instead.
        """
        recipe_path = self.args.recipe
        work_dir = getattr(self.args, 'work_dir', './otx-workspace')
        barebones = getattr(self.args, 'barebones', False)
        
        # Build otx train command
        cmd = [
            "otx", "train",
            "--config", str(recipe_path),
            "--data_root", str(self.args.data_root),
            "--work_dir", str(work_dir),
        ]
        
        # Optional device override (OTX uses 'gpu' for CUDA devices)
        if hasattr(self.args, 'device') and self.args.device:
            otx_device = "gpu" if self.args.device == "cuda" else self.args.device
            cmd.extend(["--engine.device", otx_device])
        
        # Barebones mode: disable checkpointing, progress bar, and logging
        # This reduces overhead for accurate benchmarking
        if barebones:
            self.logger.info("Barebones mode: disabling checkpointing, progress bar, and logging")
            # Disable checkpointing (no checkpoint files saved)
            cmd.extend(["--enable_checkpointing", "false"])
            # Disable progress bar for minimal overhead
            cmd.extend(["--enable_progress_bar", "false"])
            # Disable logging (set logger to false)
            cmd.extend(["--logger", "false"])
        
        self.logger.info(f"OTX command: {' '.join(cmd)}")
        return cmd

    def create_datamodule(self) -> Any:
        """Not needed for subprocess approach."""
        return None

    def create_model(self) -> Any:
        """Not needed for subprocess approach."""
        return None

    def run_single_run(
        self, 
        engine: Any,  # This is actually the command list
        model: Any, 
        datamodule: Any
    ) -> tuple[float, float, dict]:
        """Execute a single OTX training run via subprocess."""
        
        cmd = engine  # engine is the command list
        
        self.logger.info("Starting OTX training via subprocess...")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Run training and measure time
        training_time, result = self.measure_time(
            self._run_subprocess, cmd
        )
        
        self.logger.info(f"Training completed in {training_time:.2f}s")
        
        # Parse metrics from output if available
        metrics = {"return_code": result}
        
        # No separate test phase in this approach
        testing_time = 0.0
        
        return training_time, testing_time, metrics
    
    def _run_subprocess(self, cmd: list[str]) -> int:
        """Run OTX command as subprocess."""
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output go to terminal
            text=True,
        )
        return result.returncode
