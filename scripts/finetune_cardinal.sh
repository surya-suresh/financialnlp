#!/bin/bash
#SBATCH --job-name=finetune-cardinal
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

TASK="${1:-direction}"
case "$TASK" in
    direction) PAIRS=data/large/pairs_direction.jsonl ;;
    surprise) PAIRS=data/large/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch -M cardinal scripts/finetune_cardinal.sh [direction|surprise]"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID task=$TASK pairs=$PAIRS start=$(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs outputs/large/"$TASK"

python -m src.models.finetune \
    --pairs "$PAIRS" \
    --out "outputs/large/$TASK" \
    --max-length 4096 \
    --bf16

echo "Done $(date)"
