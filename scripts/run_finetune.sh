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
MIN_EPS_MARGIN="${2:-0}"
EXTRA_ARGS=""
case "$TASK" in
    direction) PAIRS=data/pairs_direction.jsonl    ;;
    surprise)  PAIRS=data/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch scripts/run_finetune.sh [direction|surprise] [min-eps-margin]"; exit 1 ;;
esac

if [[ "$MIN_EPS_MARGIN" != "0" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --min-eps-margin $MIN_EPS_MARGIN"
    OUT_DIR="outputs/${TASK}_margin${MIN_EPS_MARGIN}"
else
    OUT_DIR="outputs/$TASK"
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Task:   $TASK"
echo "EPS min:$MIN_EPS_MARGIN"
echo "Pairs:  $PAIRS"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"

module load python/3.12
module load cuda/11.8.0                  # libcudart.so.11.0 for torch 2.3.1+cu118
module load cudnn/8.7.0.84-11.8          # libcudnn.so.8
# libcupti.so.11.8 lives under CUDA extras/CUPTI, not on the default lib path.
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
source venv/bin/activate

mkdir -p logs outputs

python -m src.models.finetune \
    --pairs "$PAIRS" \
    --out   "$OUT_DIR" \
    $EXTRA_ARGS

echo "End:    $(date)"
