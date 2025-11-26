#!/bin/bash

# Helper script to submit SLURM jobs for all models
# Usage: ./submit_all_models.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# List of models to benchmark (from XPU_All-884.sh)
MODELS=(
    "Padim"
    "Dfm"
    "Patchcore"
    "Dinomaly"
    "Fre"
    "Dfkde"
    "Stfpm"
    "WinClip"
    "Fastflow"
    "VlmAd"
    "Csflow"
    "ReverseDistillation"
    "Cfa"
    "Dsr"
    "Ganomaly"
    "Supersimplenet"
    "Uflow"
    "UniNet"
)

echo "Submitting SLURM jobs for ${#MODELS[@]} models..."

for MODEL in "${MODELS[@]}"; do
    echo "Submitting job for model: $MODEL"
    sbatch slurm_benchmark.sh "$MODEL"
    # Small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: ./logs/"
