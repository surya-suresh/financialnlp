#!/bin/bash
#SBATCH --job-name=eval-cardinal
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

TASK="${1:-direction}"
MODE="${2:-adapter}"

case "$TASK" in
    direction) PAIRS=data/large/pairs_direction.jsonl ;;
    surprise) PAIRS=data/large/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch -M cardinal scripts/eval_cardinal.sh [direction|surprise] [adapter|base]"; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID task=$TASK mode=$MODE start=$(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs outputs/large/"$TASK"

if [[ "$MODE" == "base" ]]; then
    python -m src.models.evaluate \
        --pairs "$PAIRS" \
        --out "outputs/large/$TASK/results_base.json" \
        --max-length 2048 \
        --bf16
else
    python -m src.models.evaluate \
        --pairs "$PAIRS" \
        --adapter "outputs/large/$TASK/adapter" \
        --out "outputs/large/$TASK/results.json" \
        --max-length 2048 \
        --bf16
fi

echo "Done $(date)"
