#!/bin/bash

# Script to run benchmarking for multiple models sequentially
# Usage: ./run_multiple_models.sh

# Configuration
DEVICE="xpu"
CATEGORY="transistor"  # Change this to your desired category
NUM_RUNS=5
SEED=42
WAIT_TIME=20
OUTPUT_DIR="./BenchMark-V2-XPU-Barebones-884"
BATCH_SIZE=8
NUM_WORKERS=4

# List of models to benchmark (add/remove models as needed)

MODELS=(
    "Dfkde" # Fastest - only classification
    "Dfm" # Fastest + accuracy
    "Padim" # Fastest
    "WinClip" # Clip based model
    "Patchcore" # Most used
    "Fre" # Balanced
    "Stfpm"  # Student Teacher + Backup
    "Fastflow" # Good accuracy
    "Cfa"
    "Supersimplenet"
    "Dinomaly" # New SOTA. 
    "ReverseDistillation"
    # "VlmAd" # Vision-language model based method
    "Cflow"
    "Csflow"
    "Dsr"
    "Ganomaly"
    "Uflow"
    "UniNet"
)

    
    
# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if benchmark_script.py exists
if [ ! -f "benchmark_script.py" ]; then
    log_message "ERROR: benchmark_script.py not found in current directory"
    exit 1
fi

log_message "Starting benchmarking for ${#MODELS[@]} models"
log_message "Device: $DEVICE, Category: $CATEGORY, Runs: $NUM_RUNS"
log_message "Models to benchmark: ${MODELS[*]}"

# Counter for successful and failed runs
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_MODELS=()


# Run benchmark for each model
for MODEL in "${MODELS[@]}"; do
    log_message "Starting benchmark for model: $MODEL"
    
    # Run the benchmark command
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
        log_message "SUCCESS: Benchmark completed for $MODEL"
        ((SUCCESSFUL_RUNS++))
    else
        log_message "ERROR: Benchmark failed for $MODEL"
        ((FAILED_RUNS++))
        FAILED_MODELS+=("$MODEL")
    fi

done

# Summary
log_message "Benchmarking completed!"
log_message "Successful runs: $SUCCESSFUL_RUNS"
log_message "Failed runs: $FAILED_RUNS"

if [ $FAILED_RUNS -gt 0 ]; then
    log_message "Failed models: ${FAILED_MODELS[*]}"
fi

log_message "Results saved in: $OUTPUT_DIR"