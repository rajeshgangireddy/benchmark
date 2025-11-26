#!/bin/bash
#SBATCH --job-name=anomalib_benchmark
#SBATCH --output=logs/anomalib_%j_%x.out
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

# Environment setup
echo "=== Job Started: $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "SLURM Assigned GPU(s): $SLURM_JOB_GPUS"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib

# Navigate to benchmark directory
cd /Users/rgangire/workspace/code/Anomalib/benchmark/

echo "PWD: $(pwd)"

# Configuration - model name passed as first parameter
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "ERROR: Model name not provided"
    echo "Usage: sbatch slurm_benchmark.sh <model_name>"
    exit 1
fi

# Benchmark configuration (adjust these as needed)
DEVICE="cuda"
CATEGORY="transistor"
NUM_RUNS=5
SEED=42
WAIT_TIME=20
OUTPUT_DIR="./benchmark_results/BenchMark-V2-CUDA-SLURM"
BATCH_SIZE=8
NUM_WORKERS=4

echo "=== Benchmark Configuration ==="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Category: $CATEGORY"
echo "Runs: $NUM_RUNS"
echo "Output Directory: $OUTPUT_DIR"
echo "================================"

# Check if benchmark_script.py exists
if [ ! -f "benchmark_script.py" ]; then
    echo "ERROR: benchmark_script.py not found in current directory"
    exit 1
fi

# Run the benchmark
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting benchmark for model: $MODEL"

python benchmark_script.py \
    --device "$DEVICE" \
    --model_name "$MODEL" \
    --category "$CATEGORY" \
    --num_runs "$NUM_RUNS" \
    --seed "$SEED" \
    --wait_time "$WAIT_TIME" \
    --output_dir "$OUTPUT_DIR" \
    --train_batch_size "$BATCH_SIZE" \
    --eval_batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --barebones

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: Benchmark completed for $MODEL"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Benchmark failed for $MODEL"
    exit 1
fi

echo "=== Job Completed: $(date) ==="
echo "Results saved in: $OUTPUT_DIR"
