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

module load python/3.12
module load cuda/11.8.0                  # libcudart.so.11.0 for torch 2.3.1+cu118
module load cudnn/8.7.0.84-11.8          # libcudnn.so.8
# libcupti.so.11.8 lives under CUDA extras/CUPTI, not on the default lib path.
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
source venv/bin/activate

mkdir -p logs

python src/main.py \
    --max-samples 500 \
    --train-ratio 0.8 \
    --price-window 3

echo ""
echo "End time: $(date)"
echo "Results printed to logs/finance_nlp_${SLURM_JOB_ID}.out"
