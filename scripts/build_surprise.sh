#!/bin/bash
#SBATCH --job-name=build-surprise
#SBATCH --account=PAS3272
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_surprise_%j.out
#SBATCH --error=logs/build_surprise_%j.err

set -euo pipefail

echo "Job $SLURM_JOB_ID start=$(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs data/large

python src/data/build_pairs.py \
    --task surprise \
    --max-samples 8000 \
    --newest-first \
    --delay-sec 1.5 \
    --out-dir data/large

echo "Done $(date)"
