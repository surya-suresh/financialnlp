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

TASK="${1:-direction}"   # direction | surprise
MODE="${2:-base}"        # base | adapter
EXTRA_ARGS=""

case "$TASK" in
    direction)
        PAIRS=data/pairs_direction.jsonl
        ADAPTER=outputs/direction/adapter
        ;;
    surprise)
        PAIRS=data/pairs_eps_surprise.jsonl
        ADAPTER=outputs/surprise/adapter
        EXTRA_ARGS="--balance-test"
        ;;
    *)
        echo "Usage: sbatch scripts/run_prompt_eval.sh [direction|surprise] [base|adapter]"
        exit 1
        ;;
esac

case "$MODE" in
    base)    OUT="outputs/${TASK}/results_prompt_base_v3.json" ;;
    adapter) OUT="outputs/${TASK}/results_prompt_ft_v3.json"   ;;
    *)
        echo "MODE must be 'base' or 'adapter'"
        exit 1
        ;;
esac

echo "Job ID:  $SLURM_JOB_ID"
echo "Task:    $TASK   Mode: $MODE"
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

if [[ "$MODE" == "adapter" ]]; then
    python -m src.models.prompt_eval \
        --pairs          "$PAIRS" \
        --adapter        "$ADAPTER" \
        --out            "$OUT" \
        --n-shots        3 \
        --excerpt-tokens 200 \
        $EXTRA_ARGS
else
    python -m src.models.prompt_eval \
        --pairs          "$PAIRS" \
        --out            "$OUT" \
        --n-shots        3 \
        --excerpt-tokens 200 \
        $EXTRA_ARGS
fi

echo "End:     $(date)"
