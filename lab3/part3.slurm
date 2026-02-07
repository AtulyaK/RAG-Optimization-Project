#!/bin/bash
#SBATCH -A gpu                 # Account or partition name
#SBATCH -J part1_a_job         # Job name
#SBATCH -N 1                   # Number of nodes
#SBATCH --gres=gpu:2           # GPUs per node
#SBATCH --cpus-per-task=4      # CPUs per task
#SBATCH --ntasks-per-node=1    # One task per node
#SBATCH --time=01:00:00        # Max walltime
#SBATCH -o slurm_logs/part1_a_%j.out      # Standard output
#SBATCH -e slurm_logs/part1_a_%j.err      # Standard error
#SBATCH --constraint=J         # Run on J sub-cluster (48 GB GPUs)

# Unbuffered output
export PYTHONUNBUFFERED=1
export SLURM_STDOUTMODE=flush

# Load environment
module load cuda
module load anaconda/2024.02-py311
conda activate aiForge

# --- Debug info ---
echo "Job started on $(date)"
echo "Running on nodes: $SLURM_NODELIST"
echo "Using GPUs: $SLURM_GPUS"
echo "Using CPUs per task: $SLURM_CPUS_PER_TASK"
echo "------------------------------------------------"
echo "Current directory is: $(pwd)"
#conda list
pip install -q --upgrade transformers accelerate pyserini

# Export HF Token (User must set this before running or editing the script)
export HF_TOKEN="Your_HF_Token_Here"

# --- Run your Python script ---
python3 part3_a.py

echo "------------------------------------------------"
echo "Job finished on $(date)"
