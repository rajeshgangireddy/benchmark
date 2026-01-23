# OTX (OpenVINO Training Extensions) Setup

Setup guide for benchmarking OTX models.

## Installation

### Basic Installation

```bash
pip install otx
```

### With CUDA Support

```bash
pip install otx[cuda]
```

### With XPU Support (Intel GPUs)

```bash
pip install otx[xpu]
```

## Verify Installation

```bash
# Check OTX version
python -c "import otx; print(otx.__version__)"

# List available models
otx find --task DETECTION
```

## Dataset Preparation

OTX supports various dataset formats. The format is auto-detected from the data root directory.

### Supported Formats

| Task | Formats |
|------|---------|
| Classification | ImageNet, COCO, Custom folder |
| Detection | COCO, VOC, YOLO |
| Segmentation | COCO, VOC |

### Example: COCO Format

```
data/
  coco/
    annotations/
      instances_train.json
      instances_val.json
    train/
      image1.jpg
      image2.jpg
    val/
      image1.jpg
```

### Example: Classification Folder Structure

```
data/
  imagenet/
    train/
      class1/
        img1.jpg
      class2/
        img1.jpg
    val/
      class1/
        img1.jpg
```

## Running Benchmarks

### List Available Models

```bash
python otx_train_benchmark.py --list_models --task DETECTION
python otx_train_benchmark.py --list_models --task MULTI_CLASS_CLS
```

### Training Benchmark

```bash
# Detection
python otx_train_benchmark.py \
    --device cuda \
    --task DETECTION \
    --model atss_mobilenetv2 \
    --data_root ./data/coco \
    --num_runs 3

# Classification
python otx_train_benchmark.py \
    --device cuda \
    --task MULTI_CLASS_CLS \
    --model efficientnet_b0 \
    --data_root ./data/imagenet \
    --num_runs 3
```

### Inference Benchmark

First train a model or use an existing checkpoint:

```bash
python otx_inference_benchmark.py \
    --checkpoint ./otx-workspace/best_checkpoint.ckpt \
    --data_root ./data/coco \
    --num_runs 5
```

## Common Issues

### OTX Not Found

```
ImportError: No module named 'otx'
```

Solution: Install OTX with `pip install otx`

### CUDA Out of Memory

Reduce batch size:

```bash
python otx_train_benchmark.py ... --train_batch_size 4 --eval_batch_size 4
```

### Dataset Format Not Recognized

Ensure your dataset follows one of the supported formats. Check OTX documentation for format requirements.

## References

- [OTX GitHub Repository](https://github.com/open-edge-platform/training_extensions)
- [OTX Documentation](https://openvinotoolkit.github.io/training_extensions/)
