# Benchmark Repository Files Summary

This document describes the structure and purpose of files in the benchmark repository.

## Directory Structure

```
benchmark/
├── src/                           # Main source package
│   ├── __init__.py               # Package init with version
│   ├── otx_train_benchmark.py    # OTX training benchmark CLI
│   ├── otx_inference_benchmark.py # OTX inference benchmark CLI
│   ├── otx_results_to_excel.py   # Metrics export tool
│   ├── anomalib_train_benchmark.py # Anomalib training CLI
│   ├── anomalib_inference_benchmark.py # Anomalib inference CLI
│   ├── consolidate_results.py    # Results aggregation tool
│   ├── benchmarkers/             # Benchmark implementation classes
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base class
│   │   ├── otx.py               # OTX benchmark implementation
│   │   └── anomalib.py          # Anomalib benchmark implementation
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── config.py            # YAML config loading/saving
│       ├── metrics.py           # OTX metrics utilities
│       ├── statistics.py        # Statistical analysis functions
│       ├── system_info.py       # System information collection
│       └── dataset.py           # Dataset and model utilities
├── configs/                      # Benchmark config presets
│   ├── quick_test.yaml
│   ├── full_benchmark.yaml
│   ├── detection_benchmark.yaml
│   ├── inference_benchmark.yaml
│   ├── xpu_benchmark.yaml
│   └── cpu_benchmark.yaml
├── data/                         # Datasets
├── results/                      # Benchmark outputs
├── power/                        # Power monitoring tools
├── scripts/                      # Shell scripts
├── setup/                        # Setup documentation
└── deprecated/                   # Old/unused code
```

## CLI Entry Points (src/)

All CLI scripts are run as Python modules from the project root:

### OTX Benchmarks

1. **`src/otx_train_benchmark.py`** - OTX training benchmark
   ```bash
   python -m src.otx_train_benchmark --config configs/quick_test.yaml
   python -m src.otx_train_benchmark --device cuda --task MULTI_CLASS_CLS --model efficientnet_b0 --data_root ./data
   python -m src.otx_train_benchmark --list_models --task DETECTION
   ```

2. **`src/otx_inference_benchmark.py`** - OTX inference benchmark
   ```bash
   python -m src.otx_inference_benchmark --config configs/inference_benchmark.yaml --checkpoint ./model.ckpt
   python -m src.otx_inference_benchmark --checkpoint ./model.ckpt --data_root ./data --num_runs 5
   ```

3. **`src/otx_results_to_excel.py`** - Convert OTX metrics CSV to Excel
   ```bash
   python -m src.otx_results_to_excel                              # Auto-find latest run
   python -m src.otx_results_to_excel --metrics-csv /path/to/metrics.csv --output results.xlsx
   ```

### Anomalib Benchmarks (requires anomalib)

4. **`src/anomalib_train_benchmark.py`** - Anomalib training benchmark
   ```bash
   python -m src.anomalib_train_benchmark --device cuda --model_name Padim --category bottle --num_runs 5
   ```

5. **`src/anomalib_inference_benchmark.py`** - Anomalib inference benchmark
   ```bash
   python -m src.anomalib_inference_benchmark --device cuda --model Padim --category transistor --num-runs 3 --num-inferences 100
   ```

### Results Utilities

6. **`src/consolidate_results.py`** - Aggregate multiple benchmark results
   ```bash
   python -m src.consolidate_results --input-dir benchmark_results --output consolidated.xlsx
   ```

## Benchmark Implementation Classes (src/benchmarkers/)

- **`base.py`** - `BaseBenchmark` abstract class
  - Common functionality: logging, device synchronization, multi-run execution
  - Thread-safe time measurement with proper GPU sync

- **`otx.py`** - `OTXBenchmark` class
  - Implements OTX-specific training logic using OTX Engine
  - Configurable workspace directory
  - Supports all OTX task types and models

- **`anomalib.py`** - `AnomalibBenchmark` class
  - Implements Anomalib-specific training/inference
  - Supports MVTec AD dataset categories

## Utility Modules (src/utils/)

- **`config.py`** - YAML configuration management
  - `BenchmarkConfig` dataclass with nested device, data, training configs
  - `load_config()`, `save_config()` for YAML I/O
  - CLI argument merging (CLI overrides config values)

- **`metrics.py`** - OTX metrics utilities
  - `find_latest_otx_workspace()` - Locate latest training run
  - `load_otx_metrics_csv()` - Load and clean OTX metrics
  - `export_detailed_metrics_to_excel()` - Full Excel export with summary

- **`statistics.py`** - Statistical analysis
  - `summarise_results()` - Mean/std across all runs
  - `summarise_results_mlperf()` - MLPerf averaging (drops fastest/slowest)
  - `summarise_inference_results()` - Inference-specific metrics

- **`system_info.py`** - System information collection
  - CPU, GPU, memory, OS, PyTorch version details

- **`dataset.py`** - Dataset and model utilities
  - `OTX_TASK_TYPES` - Available OTX task types
  - `list_otx_models()` - List models for a task
  - `MVTEC_CATEGORIES` - MVTec AD dataset categories

## Configuration Files (configs/)

YAML config files for reproducible benchmarks:

| Config File | Purpose | Key Settings |
|-------------|---------|--------------|
| `quick_test.yaml` | Validation | 1 run, 2 epochs, minimal settings |
| `full_benchmark.yaml` | Production | 5 runs, 20 epochs, MLPerf stats |
| `detection_benchmark.yaml` | Object detection | ATSS/YOLOX models |
| `inference_benchmark.yaml` | Inference only | Checkpoint path, iterations |
| `xpu_benchmark.yaml` | Intel XPU | XPU device, bf16 precision |
| `cpu_benchmark.yaml` | CPU-only | CPU device, fp32 precision |

## Helper Scripts

- **`run_otx_benchmark.sh`** - Wrapper for OTX benchmarks with dependency handling
- **`scripts/slurm_benchmark.sh`** - SLURM cluster submission script
- **`scripts/submit_all_models.sh`** - Batch submission of all model benchmarks

## Dependencies

Required packages:
- `otx` - OpenVINO Training Extensions (for OTX benchmarks)
- `anomalib` - Anomaly detection library (optional, for Anomalib benchmarks)
- `pandas`, `openpyxl` - Excel export
- `pyyaml` - Config file parsing
- `torch`, `lightning` - Deep learning framework

## Migration from Old Structure

Old files at project root have been moved to the `src/` package:
- `benchmarker_base.py` → `src/benchmarkers/base.py`
- `otx_benchmarker.py` → `src/benchmarkers/otx.py`
- `anomalib_benchmarker.py` → `src/benchmarkers/anomalib.py`
- `otx_train_benchmark.py` → `src/otx_train_benchmark.py`
- `otx_inference_benchmark.py` → `src/otx_inference_benchmark.py`
- `utils/*.py` → `src/utils/*.py`

The old files at project root can be removed after verification.
