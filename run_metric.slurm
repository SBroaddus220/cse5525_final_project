#!/bin/bash
#SBATCH --account=PAS2912
#SBATCH --job-name=metric_job
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load CUDA module
ml cuda/11.8.0  # Ensure it matches your PyTorch installation

# Activate your virtual environment
source ./.venv/bin/activate  # Adjust path if `.venv` is in a project folder

# Run your script with the argument
python run_metric.py $1
