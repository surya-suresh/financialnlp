#!/bin/bash
# =============================================================================
# QLoRA finetune Qwen2.5-7B-Instruct on one task.
# Usage:  sbatch scripts/run_finetune.sh direction
#         sbatch scripts/run_finetune.sh surprise
# =============================================================================
#SBATCH --job-name=finetune
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

TASK="${1:-direction}"
case "$TASK" in
    direction) PAIRS=data/pairs_direction.jsonl    ;;
    surprise)  PAIRS=data/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch scripts/run_finetune.sh [direction|surprise]"; exit 1 ;;
esac

echo "Job ID: $SLURM_JOB_ID"
echo "Task:   $TASK"
echo "Pairs:  $PAIRS"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"

module load python/3.12
source venv/bin/activate

mkdir -p logs outputs

python -m src.models.finetune \
    --pairs "$PAIRS" \
    --out   "outputs/$TASK"

echo "End:    $(date)"
