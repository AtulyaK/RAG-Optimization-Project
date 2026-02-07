#!/bin/bash
#SBATCH -A gpu                 # Account or partition name
#SBATCH -J part4_job         # Job name
#SBATCH -N 1                   # Number of nodes
#SBATCH --gres=gpu:2           # GPUs per node
#SBATCH --cpus-per-task=4      # CPUs per task
#SBATCH --ntasks-per-node=1    # One task per node
#SBATCH --time=02:00:00        # Max walltime
#SBATCH -o {file_name_here} # Standard output
#SBATCH -e {file_name_here} # Standard error
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
conda list

# installation of modules not recognized from the conda environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q --upgrade transformers accelerate pyserini sentence_transformers
pip install google-genai ragas langchain-groq
# --- Run your Python script ---
python3 #TODO: insert name of Python script here

echo "------------------------------------------------"
echo "Job finished on $(date)"
