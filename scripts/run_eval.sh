#!/bin/bash
# =============================================================================
# Evaluate on the held-out chronological test split (last 10% of pairs).
# Usage:  sbatch scripts/run_eval.sh direction              # finetuned adapter
#         sbatch scripts/run_eval.sh direction base         # zero-shot base
#         sbatch scripts/run_eval.sh surprise
#         sbatch scripts/run_eval.sh surprise base
# =============================================================================
#SBATCH --job-name=eval
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

TASK="${1:-direction}"
MODE="${2:-adapter}"   # "adapter" (default) or "base"
EXTRA_ARGS=""

case "$TASK" in
    direction) PAIRS=data/pairs_direction.jsonl    ;;
    surprise)  PAIRS=data/pairs_eps_surprise.jsonl
               EXTRA_ARGS="--balance-test"          ;;
    *) echo "Usage: sbatch scripts/run_eval.sh [direction|surprise] [adapter|base]"; exit 1 ;;
esac

echo "Job ID: $SLURM_JOB_ID"
echo "Task:   $TASK   Mode: $MODE"
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

if [[ "$MODE" == "base" ]]; then
    python -m src.models.evaluate \
        --pairs "$PAIRS" \
        --out   "outputs/$TASK/results_base.json" \
        $EXTRA_ARGS
else
    python -m src.models.evaluate \
        --pairs   "$PAIRS" \
        --adapter "outputs/$TASK/adapter" \
        --out     "outputs/$TASK/results.json" \
        $EXTRA_ARGS
fi

echo "End:    $(date)"
