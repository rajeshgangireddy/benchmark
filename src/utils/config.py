# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configuration management for OTX benchmarking.
Uses YAML format (same as OTX recipes) for consistency and reproducibility.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeviceConfig:
    """Device configuration."""
    type: str = "cuda"  # cpu, cuda, xpu
    cuda_visible_devices: str = "0"


@dataclass
class DataConfig:
    """Data configuration."""
    root: str = ""
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 10
    precision: str | None = None  # null, "16", "32", "bf16-mixed"


@dataclass  
class InferenceConfig:
    """Inference benchmark configuration."""
    checkpoint: str = ""
    batch_size: int = 1
    num_inferences: int = 100
    extended_stats: bool = False


@dataclass
class RunConfig:
    """Benchmark run configuration."""
    num_runs: int = 5
    seed: int = 42
    wait_time: int = 20  # seconds between runs


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = "./benchmark_results"
    export_detailed_metrics: bool = True
    workspace_dir: str = "../training_extensions/library/otx-workspace"


@dataclass
class BenchmarkConfig:
    """
    Complete benchmark configuration.
    
    This configuration can be loaded from a YAML file for reproducibility.
    CLI arguments can override any value in the config.
    """
    # Benchmark metadata
    name: str = ""
    description: str = ""
    
    # Framework and benchmark type
    framework: str = "otx"  # otx or anomalib
    benchmark_type: str = "training"  # training or inference
    
    # Model configuration
    task: str = ""  # OTX task type (MULTI_CLASS_CLS, DETECTION, etc.)
    model: str = ""  # Model name or path to recipe YAML
    
    # Sub-configurations
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Convert nested dicts to dataclass instances if loaded from YAML."""
        if isinstance(self.device, dict):
            self.device = DeviceConfig(**self.device)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.inference, dict):
            self.inference = InferenceConfig(**self.inference)
        if isinstance(self.run, dict):
            self.run = RunConfig(**self.run)
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)


def load_config(config_path: str | Path) -> BenchmarkConfig:
    """
    Load benchmark configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        BenchmarkConfig instance.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty config file: {config_path}")
    
    return BenchmarkConfig(**config_dict)


def save_config(config: BenchmarkConfig, output_path: str | Path) -> None:
    """
    Save benchmark configuration to a YAML file.
    
    Args:
        config: BenchmarkConfig instance to save.
        output_path: Path to save the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_cli_args(config: BenchmarkConfig, args: Any) -> BenchmarkConfig:
    """
    Merge CLI arguments into a config, with CLI args taking precedence.
    
    Args:
        config: Base BenchmarkConfig.
        args: argparse.Namespace with CLI arguments.
        
    Returns:
        Updated BenchmarkConfig with CLI overrides applied.
    """
    # Map CLI args to config fields - only override if explicitly set
    cli_mappings = {
        # Top level
        'task': ('task', None),
        'model': ('model', None),
        
        # Device
        'device': ('device', 'type'),
        
        # Data
        'data_root': ('data', 'root'),
        'train_batch_size': ('data', 'train_batch_size'),
        'eval_batch_size': ('data', 'eval_batch_size'),
        'num_workers': ('data', 'num_workers'),
        
        # Training
        'max_epochs': ('training', 'max_epochs'),
        'precision': ('training', 'precision'),
        
        # Inference
        'checkpoint': ('inference', 'checkpoint'),
        'batch_size': ('inference', 'batch_size'),
        'num_inferences': ('inference', 'num_inferences'),
        'extended_stats': ('inference', 'extended_stats'),
        
        # Run
        'num_runs': ('run', 'num_runs'),
        'seed': ('run', 'seed'),
        'wait_time': ('run', 'wait_time'),
        
        # Output
        'output_dir': ('output', 'directory'),
        'export_otx_metrics': ('output', 'export_detailed_metrics'),
        'workspace_dir': ('output', 'workspace_dir'),
    }
    
    for cli_arg, (config_section, config_field) in cli_mappings.items():
        if hasattr(args, cli_arg):
            cli_value = getattr(args, cli_arg)
            # Only apply if CLI arg was explicitly provided (not None for optional args)
            if cli_value is not None:
                if config_field is None:
                    # Top-level field
                    setattr(config, config_section, cli_value)
                else:
                    # Nested field
                    section = getattr(config, config_section)
                    setattr(section, config_field, cli_value)
    
    return config


def config_to_args_namespace(config: BenchmarkConfig) -> Any:
    """
    Convert a BenchmarkConfig to an argparse.Namespace-like object.
    
    This allows the config to be used with existing code that expects args.
    
    Args:
        config: BenchmarkConfig instance.
        
    Returns:
        SimpleNamespace with flattened config values.
    """
    from types import SimpleNamespace
    
    # Apply CUDA_VISIBLE_DEVICES from config
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device.cuda_visible_devices
    
    return SimpleNamespace(
        # Core benchmark settings
        task=config.task,
        model=config.model,
        
        # Device
        device=config.device.type,
        
        # Data
        data_root=config.data.root,
        train_batch_size=config.data.train_batch_size,
        eval_batch_size=config.data.eval_batch_size,
        num_workers=config.data.num_workers,
        
        # Training
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        
        # Inference
        checkpoint=config.inference.checkpoint,
        batch_size=config.inference.batch_size,
        num_inferences=config.inference.num_inferences,
        extended_stats=config.inference.extended_stats,
        
        # Run
        num_runs=config.run.num_runs,
        seed=config.run.seed,
        wait_time=config.run.wait_time,
        
        # Output
        output_dir=config.output.directory,
        export_otx_metrics=config.output.export_detailed_metrics,
        workspace_dir=config.output.workspace_dir,
    )


def validate_training_config(config: BenchmarkConfig) -> list[str]:
    """
    Validate configuration for training benchmark.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    if not config.task:
        errors.append("task is required")
    
    if not config.model:
        errors.append("model is required")
    
    if not config.data.root:
        errors.append("data.root is required")
    elif not Path(config.data.root).exists():
        errors.append(f"data.root does not exist: {config.data.root}")
    
    if config.training.max_epochs < 1:
        errors.append("training.max_epochs must be >= 1")
    
    if config.run.num_runs < 1:
        errors.append("run.num_runs must be >= 1")
    
    if config.device.type not in ("cpu", "cuda", "xpu"):
        errors.append(f"Invalid device.type: {config.device.type}")
    
    return errors


def validate_inference_config(config: BenchmarkConfig) -> list[str]:
    """
    Validate configuration for inference benchmark.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    if not config.inference.checkpoint:
        errors.append("inference.checkpoint is required")
    elif not Path(config.inference.checkpoint).exists():
        errors.append(f"inference.checkpoint does not exist: {config.inference.checkpoint}")
    
    if not config.data.root:
        errors.append("data.root is required")
    elif not Path(config.data.root).exists():
        errors.append(f"data.root does not exist: {config.data.root}")
    
    if config.inference.batch_size < 1:
        errors.append("inference.batch_size must be >= 1")
    
    if config.run.num_runs < 1:
        errors.append("run.num_runs must be >= 1")
    
    return errors


def get_default_config(benchmark_type: str = "training") -> BenchmarkConfig:
    """
    Get a default configuration for the specified benchmark type.
    
    Args:
        benchmark_type: Either "training" or "inference".
        
    Returns:
        BenchmarkConfig with sensible defaults.
    """
    config = BenchmarkConfig(
        framework="otx",
        benchmark_type=benchmark_type,
    )
    
    if benchmark_type == "inference":
        config.run.num_runs = 3
        config.run.wait_time = 10
    
    return config
