#!/bin/bash
# =============================================================================
# RAG evaluation — retrieval-augmented generation with base or finetuned model.
# Updated for OSC Pitzer (Account: PAS3272)
# =============================================================================
#SBATCH --job-name=rag_eval
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rag_eval_%j.out
#SBATCH --error=logs/rag_eval_%j.err

set -euo pipefail

# --- Arguments ---
TASK="${1:-direction}"
MODE="${2:-adapter}"
RAG_TOP_K="${3:-2}"
RAG_CTX_TOKENS="${4:-400}"

case "$TASK" in
    direction) PAIRS=data/pairs_direction.jsonl    ;;
    surprise)  PAIRS=data/pairs_eps_surprise.jsonl ;;
    *)
        echo "Usage: sbatch scripts/run_rag_eval.sh [direction|surprise] [adapter|base]"
        exit 1
        ;;
esac

echo "========================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Task:            $TASK"
echo "Mode:            $MODE"
echo "RAG top-k:       $RAG_TOP_K"
echo "RAG ctx tokens:  $RAG_CTX_TOKENS"
echo "Pairs:           $PAIRS"
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
    echo "ERROR: Virtual environment 'venv' not found. Please create it first."
    exit 1
fi

source venv/bin/activate

# Add CUDA and CUPTI to path
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

mkdir -p logs "outputs/${TASK}"

# Copy adapter to local scratch ($TMPDIR) to avoid NFS stale-file-handle errors
# (errno 116) on OSC when reading tokenizer JSON and adapter weights.
# $TMPDIR is a fast local SSD allocated per job — reads are reliable and quick.
if [[ "$MODE" == "adapter" ]]; then
    LOCAL_ADAPTER="$TMPDIR/adapter_${TASK}"
    echo "Copying adapter to local scratch: $LOCAL_ADAPTER"
    mkdir -p "$LOCAL_ADAPTER"
    cp -r "outputs/${TASK}/adapter/." "$LOCAL_ADAPTER/"
    echo "Adapter copy done."
fi

# --- Execution ---
if [[ "$MODE" == "base" ]]; then
    echo "Running in BASE mode..."
    python -m src.models.evaluate \
        --pairs              "$PAIRS" \
        --out                "outputs/${TASK}/results_rag_base.json" \
        --rag \
        --rag-top-k          "$RAG_TOP_K" \
        --rag-context-tokens "$RAG_CTX_TOKENS"
else
    echo "Running in ADAPTER mode..."
    python -m src.models.evaluate \
        --pairs              "$PAIRS" \
        --adapter            "$LOCAL_ADAPTER" \
        --out                "outputs/${TASK}/results_rag.json" \
        --rag \
        --rag-top-k          "$RAG_TOP_K" \
        --rag-context-tokens "$RAG_CTX_TOKENS"
fi

echo "========================================================"
echo "End Time:        $(date)"
echo "========================================================"
