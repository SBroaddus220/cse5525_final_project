#!/bin/bash
#SBATCH --account=PAS2912
#SBATCH --job-name=cse5525_final_project_compute_metrics_005
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --mem=16gb
#SBATCH --gpus-per-node=1
#SBATCH --output=compute_metrics.out
#SBATCH --error=compute_metrics.err

# Load CUDA module
ml cuda/11.8.0  # Ensure it matches your PyTorch installation

# Activate your virtual environment
source ./.venv/bin/activate  # Adjust path if `.venv` is in a project folder

# Run your Python script
python ./main.py