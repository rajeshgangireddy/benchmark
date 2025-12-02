# Anomalib Benchmarking Tool (WIP) 
This is not an official benchmarking tool. This is created for my personal testing of some of the anomalib models on various systems I have. 

Benchmarking scripts for Anomalib models across different hardware platforms (CPU, CUDA GPU, Intel XPU).

## Requirements

- Tested with Python 3.12.3 (recommended)
- PyTorch 2.9
- Anomalib 2.2.0
- Additional dependencies in `requirements.txt`

## Setup

Choose the appropriate setup guide based on your hardware:

- **CUDA (NVIDIA GPUs)**: [CUDA.md](setup/CUDA.md)
- **Intel XPU (Intel GPUs)**: [IntelXPU.md](setup/IntelXPU.md)
- **Jetson Orin**: [JetsonOrin.md](setup/JetsonOrin.md)

## Scripts

### 1. Training Benchmark (`train_benchmark.py`)

Benchmarks the training and testing phases of Anomalib models.

#### Common Examples

**Benchmark on CPU with 3 runs:**
```bash
python train_benchmark.py --device cpu --num_runs 3 --model_name Patchcore --category transistor
```

**Benchmark on Intel XPU:**
```bash
python train_benchmark.py --device xpu --model_name Patchcore --category bottle --num_runs 5
```

**Full configuration example:**
```bash
python train_benchmark.py \
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

#### Arguments

- `--device`: Device to run benchmark on. Choices: `cpu`, `cuda`, `xpu`
- `--num_runs`: Number of training+testing runs (default: 5)
  - **Note:** MLPerf-style summary requires at least 3 runs
- `--seed`: Starting random seed for reproducibility (default: 42)
- `--wait_time`: Seconds to wait between runs (default: 20)
- `--output_dir`: Directory to save results (default: `./benchmark_results`)
- `--max_epochs`: Maximum training epochs (default: 20)
- `--barebones`: Enable barebones mode (disables logging, progress bars, and checkpointing)
- `--model_name`: Anomalib model name (required)
- `--category`: MVTecAD category (required)
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 32)
- `--num_workers`: Data loading workers (default: 8)

#### Output

Results are saved as an Excel file with the following sheets:

1. **System Info**: Hardware and software configuration
2. **Benchmark Results Raw**: Individual run metrics and timings
3. **Summary**: Mean performance across all runs
4. **Summary MLPERF**: MLPerf-style summary (drops fastest/slowest runs)
   - Only generated if `num_runs >= 3`

**Filename Format:**
```
BM_{device}_{model_name}_{category}_runs-{num_runs}_seed-{seed}_{timestamp}.xlsx
```

Example: `BM_cuda_Padim_transistor_runs-5_seed-42_20251104-143022.xlsx`

---

### 2. Inference Benchmark (`inference_benchmark.py`)

Benchmarks inference performance comparing PyTorch and OpenVINO (FP32, FP16, INT8) formats.

#### Common Examples

**Benchmark single model:**
```bash
python inference_benchmark.py --device cuda --model Padim --num-inferences 100 --category transistor
```

**Benchmark multiple models:**
```bash
python inference_benchmark.py --device cuda --model Padim Patchcore EfficientAd --num-inferences 100 --category transistor
```

**Skip training and use existing exported models:**
```bash
python inference_benchmark.py --device cpu --model Padim --skip-training --model-path ./results --num-inferences 50
```

#### Arguments

- `--device`: Device to run inference on. Choices: `cuda`, `cpu`, `CPU`, `GPU` (default: `cuda`)
  - For PyTorch: `cuda` or `cpu`
  - For OpenVINO: `GPU` or `CPU`
- `--model`: Model(s) to benchmark. Can specify multiple (default: `Padim`)
- `--num-inferences`: Number of inference runs (default: 100)
- `--category`: MVTec AD category (default: `transistor`)
- `--skip-training`: Skip training and use existing exported models
- `--model-path`: Path to existing model directory (used with `--skip-training`)
- `--output`: Output Excel file path (default: auto-generated with timestamp)

#### Output

Results are saved as an Excel file with the following sheets:

1. **Summary**: Cross-model performance comparison
2. **{Model Name}** (one sheet per model): Detailed results including:
   - Configuration
   - Benchmark results (Total, Avg, Min, Max times, FPS, Export time)
   - Speedup comparison vs PyTorch

**Filename Format:**
```
multi_model_benchmark_results_{timestamp}.xlsx
```

Example: `multi_model_benchmark_results_20251201_143022.xlsx`

---

## Additional Information

---

## Additional Information

### Error Handling

- Results are automatically backed up as JSON if Excel export fails
- Partial results are preserved if benchmarking is interrupted

