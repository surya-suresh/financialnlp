#!/bin/bash
#SBATCH --job-name=build-pairs
#SBATCH --account=PAS3272
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_pairs_%j.out
#SBATCH --error=logs/build_pairs_%j.err

set -euo pipefail

echo "Job $SLURM_JOB_ID start=$(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs data

python src/data/build_pairs.py --max-samples 12000 --out-dir data

echo "Done $(date)"
