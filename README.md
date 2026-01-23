# ML Model Benchmarking Tool

Benchmarking scripts for Anomalib and OTX (OpenVINO Training Extensions) models across different hardware platforms (CPU, CUDA GPU, Intel XPU).

## Supported Frameworks

- **Anomalib**: Anomaly detection models (Padim, Patchcore, EfficientAd, etc.)
- **OTX**: Classification, detection, and segmentation models from Intel's training_extensions

## Requirements

- Python 3.12+ (recommended)
- PyTorch 2.9+
- Anomalib 2.2.0 (for anomaly detection benchmarks)
- OTX (for OTX benchmarks) - `pip install otx`

## Setup

Choose the appropriate setup guide based on your hardware:

- **CUDA (NVIDIA GPUs)**: [Cuda.md](setup/Cuda.md)
- **Intel XPU (Intel GPUs)**: [IntelXPU.md](setup/IntelXPU.md)
- **Jetson Orin**: [JetsonOrin.md](setup/JetsonOrin.md)
- **OTX Setup**: [OTX.md](setup/OTX.md)

---

## Anomalib Benchmarks

### Training Benchmark (`anomalib_train_benchmark.py`)

Benchmarks training and testing phases of Anomalib models.

```bash
# Basic usage
python anomalib_train_benchmark.py --device cuda --model_name Padim --category transistor

# Full configuration
python anomalib_train_benchmark.py \
    --device cuda \
    --model_name Padim \
    --category transistor \
    --num_runs 5 \
    --max_epochs 20 \
    --precision 16 \
    --barebones
```

#### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | cpu, cuda, xpu | required |
| `--model_name` | Anomalib model name | required |
| `--category` | MVTecAD category | required |
| `--num_runs` | Number of runs (>=3 for MLPerf) | 5 |
| `--max_epochs` | Training epochs | 20 |
| `--precision` | 16, 32, bf16-mixed | framework default |
| `--barebones` | Disable logging/checkpoints | false |

### Inference Benchmark (`inference_benchmark.py`)

Benchmarks PyTorch and OpenVINO inference (FP32, FP16, INT8).

```bash
python inference_benchmark.py --device cuda --model Padim Patchcore --num-inferences 100 --category transistor
```

---

## OTX Benchmarks

### Training Benchmark (`otx_train_benchmark.py`)

Benchmarks OTX model training across classification, detection, and segmentation tasks.

```bash
# List available models for a task
python otx_train_benchmark.py --list_models --task DETECTION

# Benchmark a detection model
python otx_train_benchmark.py \
    --device cuda \
    --task DETECTION \
    --model atss_mobilenetv2 \
    --data_root ./data/coco \
    --num_runs 5 \
    --max_epochs 10

# Benchmark a classification model
python otx_train_benchmark.py \
    --device xpu \
    --task MULTI_CLASS_CLS \
    --model efficientnet_b0 \
    --data_root ./data/imagenet

# Use a recipe file
python otx_train_benchmark.py \
    --device cuda \
    --task DETECTION \
    --model ./recipes/yolox.yaml \
    --data_root ./data
```

#### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | cpu, cuda, xpu | cuda |
| `--task` | OTX task type (see below) | required |
| `--model` | Model name or recipe path | required |
| `--data_root` | Dataset directory | required |
| `--num_runs` | Number of runs | 5 |
| `--max_epochs` | Training epochs | 10 |
| `--precision` | 16, 32, bf16-mixed | framework default |
| `--list_models` | List models for task | - |

#### Supported OTX Tasks

- `MULTI_CLASS_CLS` - Multi-class classification
- `MULTI_LABEL_CLS` - Multi-label classification
- `H_LABEL_CLS` - Hierarchical label classification
- `DETECTION` - Object detection
- `ROTATED_DETECTION` - Rotated object detection
- `INSTANCE_SEGMENTATION` - Instance segmentation
- `SEMANTIC_SEGMENTATION` - Semantic segmentation
- `KEYPOINT_DETECTION` - Keypoint detection

### Inference Benchmark (`otx_inference_benchmark.py`)

Wraps OTX's built-in `engine.benchmark()` with multi-run averaging.

```bash
python otx_inference_benchmark.py \
    --checkpoint ./otx-workspace/best_checkpoint.ckpt \
    --data_root ./data/coco \
    --num_runs 5 \
    --batch_size 1 \
    --num_inferences 100
```

#### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to .ckpt file | required |
| `--data_root` | Dataset directory | required |
| `--device` | cpu, cuda, xpu | cuda |
| `--num_runs` | Number of runs | 3 |
| `--num_inferences` | Iterations per run | 100 |
| `--batch_size` | Inference batch size | 1 |

---

## Output Format

All benchmarks save results as Excel files with multiple sheets:

1. **System Info**: Hardware/software configuration
2. **Benchmark Config**: Run configuration (OTX only)
3. **Raw Results**: Individual run data
4. **Summary**: Mean and standard deviation across runs
5. **Summary MLPerf**: MLPerf-style summary (drops fastest/slowest, requires >=3 runs)

### Filename Patterns

```
# Anomalib training
BM_{device}_{model}_{category}_runs-{n}_seed-{s}_{timestamp}.xlsx

# OTX training
OTX_BM_{device}_{task}_{model}_runs-{n}_seed-{s}_{timestamp}.xlsx

# OTX inference
OTX_INF_BM_{device}_{checkpoint}_runs-{n}_{timestamp}.xlsx
```

---

## Statistical Methods

### Standard Summary
- Mean and standard deviation across all runs

### MLPerf Summary
- Drops fastest and slowest runs
- Reports mean of remaining runs
- Requires minimum 3 runs
- More robust against outliers

---

## Error Handling

- Results are backed up as JSON if Excel export fails
- Partial results preserved if benchmark is interrupted
- Version mismatches logged as warnings

---

## Legacy Scripts

The original `train_benchmark.py` and `benchmarker.py` are preserved for backwards compatibility but the refactored versions (`anomalib_train_benchmark.py`, `anomalib_benchmarker.py`) are recommended.

