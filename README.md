# Anomalib Benchmarking Tool (WIP) 
This is not an official benchmarking tool. This is created for my personal testing of some of the anomalib models on various systems I have. 

Benchmarking script for Anomalib models across different hardware platforms (CPU, CUDA GPU, Intel XPU).

## Requirements

- Tested with Python 3.12.3 (recommended)
- PyTorch 2.9
- Anomalib 2.2.0
- Additional dependencies in `requirements.txt`

## Setup 
For cuda and  xpu devices, the setup is straight forward (will be updated soon)
For setting up on Jetson Orin, see : [JetsonOrin.md](setup/JetsonOrin.md)

## Usage

### Common Examples

**Benchmark on CPU with 3 runs:**
```bash
python benchmark_script.py --device cpu --num_runs 3 --model_name Patchcore --category transistor
```

**Benchmark on Intel XPU :**
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

```bash
- `--device`: Device to run benchmark on. Choices: `cpu`, `cuda`, `xpu`

- `--num_runs`: Number of training+testing runs (default: 5)
  - **Note:** MLPerf-style summary requires at least 3 runs
- `--seed`: Starting random seed for reproducibility (default: 42)
- `--wait_time`: Seconds to wait between runs (default: 20)
- `--output_dir`: Directory to save results (default: `./benchmark_results`)


- `--max_epochs`: Maximum training epochs (default: 20)
- `--save_checkpoint_during_training`: Enable checkpoint saving (may slow training)


- `--model_name`: Anomalib model name (default: `Padim`)
- `--category`: MVTecAD category (default: `transistor`)
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 32)
- `--num_workers`: Data loading workers (default: 8)

```
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

