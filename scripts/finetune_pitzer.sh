#!/bin/bash
#SBATCH --job-name=finetune-pitzer
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_pitzer_%j.out
#SBATCH --error=logs/finetune_pitzer_%j.err

set -euo pipefail

TASK="${1:-direction}"
case "$TASK" in
    direction) PAIRS=data/large/pairs_direction.jsonl ;;
    surprise) PAIRS=data/large/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch -M pitzer scripts/finetune_pitzer.sh [direction|surprise]"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID task=$TASK pairs=$PAIRS start=$(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs outputs/large/"$TASK"

python -m src.models.finetune \
    --pairs "$PAIRS" \
    --out "outputs/large/$TASK" \
    --max-length 2048

echo "Done $(date)"
