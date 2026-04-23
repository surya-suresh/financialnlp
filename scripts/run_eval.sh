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
MODE="${2:-adapter}"              # adapter | base
VIEWS="${3:-head_tail}"           # comma-separated: head_tail,tail_heavy,front,prepared,qa
BALANCE_TEST="${4:-false}"        # true | false
MIN_EPS_MARGIN="${5:-0}"
ADAPTER_OVERRIDE="${6:-}"
EXTRA_ARGS=""

case "$TASK" in
    direction) PAIRS=data/pairs_direction.jsonl    ;;
    surprise)  PAIRS=data/pairs_eps_surprise.jsonl ;;
    *) echo "Usage: sbatch scripts/run_eval.sh [direction|surprise] [adapter|base] [views] [balance-test] [min-eps-margin]"; exit 1 ;;
esac

echo "Job ID: $SLURM_JOB_ID"
echo "Task:   $TASK   Mode: $MODE"
echo "Views:  $VIEWS"
echo "Balance:$BALANCE_TEST"
echo "EPS min:$MIN_EPS_MARGIN"
echo "Adapter:${ADAPTER_OVERRIDE:-outputs/$TASK/adapter}"
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

if [[ "$BALANCE_TEST" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --balance-test"
fi

if [[ "$MIN_EPS_MARGIN" != "0" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --min-eps-margin $MIN_EPS_MARGIN"
fi

SAFE_VIEWS="${VIEWS//,/_}"
if [[ "$VIEWS" == "head_tail" && "$MIN_EPS_MARGIN" == "0" && "$BALANCE_TEST" == "false" ]]; then
    BASE_OUT="outputs/$TASK/results"
else
    BASE_OUT="outputs/$TASK/results_${SAFE_VIEWS}_margin${MIN_EPS_MARGIN}"
    if [[ "$BALANCE_TEST" == "true" ]]; then
        BASE_OUT="${BASE_OUT}_balanced"
    fi
fi

if [[ "$MODE" == "base" ]]; then
    python -m src.models.evaluate \
        --pairs "$PAIRS" \
        --out   "${BASE_OUT}_base.json" \
        --views "$VIEWS" \
        $EXTRA_ARGS
else
    ADAPTER_DIR="${ADAPTER_OVERRIDE:-outputs/$TASK/adapter}"
    python -m src.models.evaluate \
        --pairs   "$PAIRS" \
        --adapter "$ADAPTER_DIR" \
        --out     "${BASE_OUT}.json" \
        --views   "$VIEWS" \
        $EXTRA_ARGS
fi

echo "End:    $(date)"
