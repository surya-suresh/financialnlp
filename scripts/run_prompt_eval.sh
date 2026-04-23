#!/bin/bash
# =============================================================================
# k-shot prompting evaluation (base or finetuned model).
#
# Usage:
#   sbatch scripts/run_prompt_eval.sh direction base
#   sbatch scripts/run_prompt_eval.sh direction adapter
#   sbatch scripts/run_prompt_eval.sh surprise  base
#   sbatch scripts/run_prompt_eval.sh surprise  adapter
# =============================================================================
#SBATCH --job-name=prompt_eval
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prompt_eval_%j.out
#SBATCH --error=logs/prompt_eval_%j.err

set -euo pipefail

TASK="${1:-direction}"        # direction | surprise
MODE="${2:-base}"             # base | adapter
SEED="${3:-42}"
BALANCE_TEST="${4:-false}"    # true | false
MIN_EPS_MARGIN="${5:-0}"
ADAPTER_OVERRIDE="${6:-}"
CONTEXT="${7:-false}"         # true | false  — prepend non-transcript data block
EXTRA_ARGS=""

case "$TASK" in
    direction)
        PAIRS=data/pairs_direction.jsonl
        ADAPTER=outputs/direction/adapter
        ;;
    surprise)
        PAIRS=data/pairs_eps_surprise.jsonl
        ADAPTER=outputs/surprise/adapter
        ;;
    *)
        echo "Usage: sbatch scripts/run_prompt_eval.sh [direction|surprise] [base|adapter]"
        exit 1
        ;;
esac

OUT_SUFFIX="seed${SEED}"
if [[ "$MIN_EPS_MARGIN" != "0" ]]; then
    OUT_SUFFIX="${OUT_SUFFIX}_margin${MIN_EPS_MARGIN}"
fi
if [[ "$BALANCE_TEST" == "true" ]]; then
    OUT_SUFFIX="${OUT_SUFFIX}_balanced"
fi
if [[ "$CONTEXT" == "true" ]]; then
    OUT_SUFFIX="${OUT_SUFFIX}_ctx"
fi

case "$MODE" in
    base)    OUT="outputs/${TASK}/results_prompt_base_v4_${OUT_SUFFIX}.json" ;;
    adapter) OUT="outputs/${TASK}/results_prompt_ft_v4_${OUT_SUFFIX}.json"   ;;
    *)
        echo "MODE must be 'base' or 'adapter'"
        exit 1
        ;;
esac

echo "Job ID:  $SLURM_JOB_ID"
echo "Task:    $TASK   Mode: $MODE"
echo "Seed:    $SEED"
echo "Balance: $BALANCE_TEST"
echo "EPS min: $MIN_EPS_MARGIN"
echo "Context: $CONTEXT"
echo "Adapter: ${ADAPTER_OVERRIDE:-$ADAPTER}"
echo "Pairs:   $PAIRS"
echo "Out:     $OUT"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:   $(date)"

module load python/3.12
module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
source venv/bin/activate

mkdir -p logs outputs/"$TASK"

if [[ "$BALANCE_TEST" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --balance-test"
fi

if [[ "$MIN_EPS_MARGIN" != "0" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --min-eps-margin $MIN_EPS_MARGIN"
fi

if [[ "$MODE" == "adapter" ]]; then
    ADAPTER_DIR="${ADAPTER_OVERRIDE:-$ADAPTER}"
    python -m src.models.prompt_eval \
        --pairs          "$PAIRS" \
        --adapter        "$ADAPTER_DIR" \
        --out            "$OUT" \
        --n-shots        4 \
        --excerpt-tokens 150 \
        --max-length     3072 \
        --context \
        --seed           "$SEED" \
        $EXTRA_ARGS
else
    python -m src.models.prompt_eval \
        --pairs          "$PAIRS" \
        --out            "$OUT" \
        --n-shots        4 \
        --excerpt-tokens 150 \
        --max-length     3072 \
        --context \
        --seed           "$SEED" \
        $EXTRA_ARGS
fi

echo "End:     $(date)"
