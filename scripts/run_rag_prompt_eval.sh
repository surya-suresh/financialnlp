#!/bin/bash
# =============================================================================
# RAG + few-shot prompting evaluation.
# Updated for OSC Pitzer (Account: PAS3272)
# =============================================================================
#SBATCH --job-name=rag_prompt_eval
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rag_prompt_eval_%j.out
#SBATCH --error=logs/rag_prompt_eval_%j.err

set -euo pipefail

# --- Arguments ---
TASK="${1:-direction}"
MODE="${2:-base}"
SEED="${3:-42}"
RAG_TOP_K="${4:-2}"
RAG_CTX_TOKENS="${5:-400}"

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
        echo "Usage: sbatch scripts/run_rag_prompt_eval.sh [direction|surprise] [base|adapter]"
        exit 1
        ;;
esac

case "$MODE" in
    base)    OUT="outputs/${TASK}/results_prompt_rag_base_v4_seed${SEED}.json" ;;
    adapter) OUT="outputs/${TASK}/results_prompt_rag_ft_v4_seed${SEED}.json"   ;;
    *)
        echo "MODE must be 'base' or 'adapter'"
        exit 1
        ;;
esac

echo "========================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Task:            $TASK"
echo "Mode:            $MODE"
echo "Seed:            $SEED"
echo "RAG top-k:       $RAG_TOP_K"
echo "RAG ctx tokens:  $RAG_CTX_TOKENS"
echo "Pairs:           $PAIRS"
echo "Out:             $OUT"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start Time:      $(date)"
echo "========================================================"

# --- Environment Setup ---
module purge
module load python/3.12
module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8

# Ensure venv exists before activating
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found."
    exit 1
fi

source venv/bin/activate

# Add CUDA and CUPTI to path
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

mkdir -p logs "outputs/${TASK}"

# Copy adapter to local scratch ($TMPDIR) to avoid NFS stale-file-handle errors
# (errno 116) on OSC when reading tokenizer JSON and adapter weights.
if [[ "$MODE" == "adapter" ]]; then
    LOCAL_ADAPTER="$TMPDIR/adapter_${TASK}"
    echo "Copying adapter to local scratch: $LOCAL_ADAPTER"
    mkdir -p "$LOCAL_ADAPTER"
    cp -r "$ADAPTER/." "$LOCAL_ADAPTER/"
    echo "Adapter copy done."
    ADAPTER="$LOCAL_ADAPTER"
fi

# --- Execution ---
if [[ "$MODE" == "adapter" ]]; then
    echo "Running PROMPT + RAG in ADAPTER mode..."
    python -m src.models.prompt_eval \
        --pairs              "$PAIRS" \
        --adapter            "$ADAPTER" \
        --out                "$OUT" \
        --n-shots            4 \
        --excerpt-tokens     150 \
        --max-length         3072 \
        --seed               "$SEED" \
        --rag \
        --rag-top-k          "$RAG_TOP_K" \
        --rag-context-tokens "$RAG_CTX_TOKENS"
else
    echo "Running PROMPT + RAG in BASE mode..."
    python -m src.models.prompt_eval \
        --pairs              "$PAIRS" \
        --out                "$OUT" \
        --n-shots            4 \
        --excerpt-tokens     150 \
        --max-length         3072 \
        --seed               "$SEED" \
        --rag \
        --rag-top-k          "$RAG_TOP_K" \
        --rag-context-tokens "$RAG_CTX_TOKENS"
fi

echo "========================================================"
echo "End Time:        $(date)"
echo "========================================================"
