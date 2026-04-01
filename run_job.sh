#!/bin/bash
# =============================================================================
# Finance NLP -- FinBERT sentiment -> stock direction prediction
# =============================================================================
#SBATCH --job-name=finance-nlp
#SBATCH --account=PAS3272
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/finance_nlp_%j.out
#SBATCH --error=logs/finance_nlp_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo ""

module load miniconda3/24.1.2-py310
source venv/bin/activate

mkdir -p logs

python src/main.py \
    --max-samples 500 \
    --train-ratio 0.8 \
    --price-window 3

echo ""
echo "End time: $(date)"
echo "Results printed to logs/finance_nlp_${SLURM_JOB_ID}.out"
