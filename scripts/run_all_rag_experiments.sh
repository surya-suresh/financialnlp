#!/bin/bash
# =============================================================================
# Submit all 8 RAG experiment conditions to SLURM in one shot.
# Updated for OSC Pitzer (Account: PAS3272)
#
# Usage:
#   bash scripts/run_all_rag_experiments.sh
# =============================================================================

set -euo pipefail

# --- Configuration ---
RAG_TOP_K="${RAG_TOP_K:-2}"
RAG_CTX_TOKENS="${RAG_CTX_TOKENS:-400}"
SEED="${SEED:-42}"
OSC_ACCOUNT="PAS3272"

echo "========================================================"
echo "Submitting 8 RAG experiment conditions"
echo "  Account         : $OSC_ACCOUNT"
echo "  RAG top-k       : $RAG_TOP_K"
echo "  RAG ctx tokens  : $RAG_CTX_TOKENS"
echo "  Prompting seed  : $SEED"
echo "========================================================"

# Guard: check that sbatch is available (OSC cluster)
if ! command -v sbatch &>/dev/null; then
    echo "ERROR: sbatch not found. Run this script on an OSC login node."
    exit 1
fi

# --- direction ---------------------------------------------------------------

# [1] direction + base + RAG
JOB1=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_eval.sh \
       direction base "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[1/8] direction + base        + RAG             → job $JOB1"

# [2] direction + base + RAG + prompting
JOB2=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_prompt_eval.sh \
       direction base "$SEED" "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[2/8] direction + base        + RAG + prompting → job $JOB2"

# [3] direction + finetuned + RAG
JOB3=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_eval.sh \
       direction adapter "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[3/8] direction + finetuned   + RAG             → job $JOB3"

# [4] direction + finetuned + RAG + prompting
JOB4=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_prompt_eval.sh \
       direction adapter "$SEED" "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[4/8] direction + finetuned   + RAG + prompting → job $JOB4"

# --- surprise ----------------------------------------------------------------

# [5] surprise + base + RAG
JOB5=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_eval.sh \
       surprise base "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[5/8] surprise  + base        + RAG             → job $JOB5"

# [6] surprise + base + RAG + prompting
JOB6=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_prompt_eval.sh \
       surprise base "$SEED" "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[6/8] surprise  + base        + RAG + prompting → job $JOB6"

# [7] surprise + finetuned + RAG
JOB7=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_eval.sh \
       surprise adapter "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[7/8] surprise  + finetuned   + RAG             → job $JOB7"

# [8] surprise + finetuned + RAG + prompting
JOB8=$(sbatch --parsable --account="$OSC_ACCOUNT" scripts/run_rag_prompt_eval.sh \
       surprise adapter "$SEED" "$RAG_TOP_K" "$RAG_CTX_TOKENS")
echo "[8/8] surprise  + finetuned   + RAG + prompting → job $JOB8"

echo ""
echo "All 8 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/rag_eval_<JOB_ID>.out"
echo ""
echo "Expected output files:"
echo "  outputs/direction/results_rag_base.json"
echo "  outputs/direction/results_prompt_rag_base_v4_seed${SEED}.json"
echo "  outputs/direction/results_rag.json"
echo "  outputs/direction/results_prompt_rag_ft_v4_seed${SEED}.json"
echo "  outputs/surprise/results_rag_base.json"
echo "  outputs/surprise/results_prompt_rag_base_v4_seed${SEED}.json"
echo "  outputs/surprise/results_rag.json"
echo "  outputs/surprise/results_prompt_rag_ft_v4_seed${SEED}.json"
