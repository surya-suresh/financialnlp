#!/bin/bash
# =============================================================================
# Build labeled (input, output) JSONL pairs for both finetuning tasks.
# Wraps src/data/build_pairs.py as a Slurm job (CPU-only, network-bound).
# Outputs:
#     data/pairs_direction.jsonl
#     data/pairs_eps_surprise.jsonl
# =============================================================================
#SBATCH --job-name=build-pairs
#SBATCH --account=PAS3272
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/build_pairs_%j.out
#SBATCH --error=logs/build_pairs_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

module load python/3.12
# `datasets` transitively imports torch, which needs libcudart.so.11.0 even on
# CPU nodes.  Load the CUDA module so the dynamic loader can find the library;
# no GPU is used.
module load cuda/11.8.0
source venv/bin/activate

mkdir -p logs data

python src/data/build_pairs.py --max-samples 5000 --out-dir data

echo "End:    $(date)"
