# Anomalib Benchmarking Tool (WIP) 

Benchmarking script for Anomalib models across different hardware platforms (CPU, CUDA GPU, Intel XPU).

## Requirements

- Python 3.12.3 (recommended)
- PyTorch 2.9
- Anomalib 2.2.0
- Additional dependencies in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

Run a benchmark on CUDA GPU with default settings:

```bash
python benchmark_script.py --device cuda
```

### Common Examples

**Benchmark on CPU with 3 runs:**
```bash
python benchmark_script.py --device cpu --num_runs 3
```

**Benchmark on Intel XPU with custom model and category:**
```bash
python benchmark_script.py --device xpu --model_name Patchcore --category bottle --num_runs 5
```

**Full configuration example:**
```bash
python benchmark_script.py \
    --device cuda \
    --model_name Padim \
    --category transistor \
    --num_runs 5 \
    --seed 42 \
    --max_epochs 20 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_workers 8 \
    --output_dir ./my_results \
    --wait_time 30
```

## Arguments

### Required Arguments
- `--device`: Device to run benchmark on. Choices: `cpu`, `cuda`, `xpu`

### Benchmark Configuration
- `--num_runs`: Number of training+testing runs (default: 5)
  - **Note:** MLPerf-style summary requires at least 3 runs
- `--seed`: Starting random seed for reproducibility (default: 42)
- `--wait_time`: Seconds to wait between runs (default: 20)
- `--output_dir`: Directory to save results (default: `./benchmark_results`)

### Training Configuration
- `--max_epochs`: Maximum training epochs (default: 20)
- `--save_checkpoint_during_training`: Enable checkpoint saving (may slow training)

### Data & Model Configuration
- `--model_name`: Anomalib model name (default: `Padim`)
- `--category`: MVTecAD category (default: `transistor`)
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 32)
- `--num_workers`: Data loading workers (default: 8)

## Output

Results are saved as an Excel file with the following sheets:

1. **System Info**: Hardware and software configuration
2. **Benchmark Results Raw**: Individual run metrics and timings
3. **Summary**: Mean performance across all runs
4. **Summary MLPERF**: MLPerf-style summary (drops fastest/slowest runs)
   - Only generated if `num_runs >= 3`

### Output Filename Format
```
benchmark_{model_name}_{category}_runs{num_runs}_seed{seed}_{timestamp}.xlsx
```

Example: `benchmark_Padim_transistor_runs5_seed42_20251104-143022.xlsx`

## Error Handling

- Results are automatically backed up as JSON if Excel export fails
- Partial results are preserved if benchmarking is interrupted
- Comprehensive logging in `benchmark.log`

## Platform-Specific Setup

### Intel XPU (Lunar Lake, Arrow Lake, etc.)
```bash
# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch
```

### NVIDIA CUDA GPU
```bash
# Ensure CUDA toolkit and drivers are installed
nvidia-smi  # Verify GPU availability
```

### CPU
No additional setup required. Recommended for testing or CPU-optimized models.

## Examples by Use Case

### Quick Test (2 runs)
```bash
python benchmark_script.py --device cpu --num_runs 2 --max_epochs 5
```
*Note: MLPerf summary will be skipped*

### Production Benchmark (10 runs)
```bash
python benchmark_script.py --device cuda --num_runs 10 --wait_time 60
```

### Memory-Constrained Environment
```bash
python benchmark_script.py --device cuda --train_batch_size 16 --eval_batch_size 16 --num_workers 4
```

## Troubleshooting

**Issue:** "MLPerf summary requires at least 3 runs"
- **Solution:** Increase `--num_runs` to 3 or more, or ignore the warning

**Issue:** Out of memory during training
- **Solution:** Reduce `--train_batch_size` and `--eval_batch_size`

**Issue:** Output directory permission error
- **Solution:** Use a different `--output_dir` or adjust permissions

## Contributing

This is an official benchmarking tool. Please report issues or suggestions through the project's issue tracker.
