#!/bin/bash
#SBATCH --account=PAS2912
#SBATCH --job-name=hello_world_test
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --gpus-per-node=1

# Load CUDA module
ml cuda/11.8.0  # Ensure it matches your PyTorch installation

# Activate your virtual environment
source ./.venv/bin/activate  # Adjust path if `.venv` is in a project folder

# Run your Python script
python ./main.py