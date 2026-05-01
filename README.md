# Finance NLP

Surya Suresh, Dheeraj Gosula, Nikhil Kumar

QLoRA fine-tuning of Qwen2.5-7B-Instruct on S&P 500 earnings call transcripts for two binary tasks:

1. Direction: stock up or down over the three trading days after the call.
2. Surprise: EPS beat or miss vs. analyst consensus.

The full report with results and analysis is at [docs/final_report.pdf](docs/final_report.pdf).

There is also an older FinBERT + logistic regression baseline in [src/main.py](src/main.py).

## Layout

```
financialnlp/
  src/
    data/         loader.py, preprocessor.py, build_pairs.py
    features/     FinBERT sentiment + extractor (baseline)
    models/       finetune.py, evaluate.py, classifier.py (baseline)
    retrieval/    build_index.py, embed_utils.py, evaluate_rag.py, fewshot.py, retriever.py
    utils/        metrics.py
    main.py       baseline pipeline (FinBERT + logreg)
  scripts/        Slurm job scripts (build, finetune, eval per cluster)
  outputs/
    small/        adapters + results from the original 2k-pair dataset
    large/        adapters + results from the larger 11k/8k-pair dataset
  docs/           proposal, final_report (.tex and .pdf)
  data/           generated locally, not in repo (see Pipeline)
  logs/           generated locally, not in repo
```

## Setup

```
bash setup.sh
source venv/bin/activate
```

Builds a Python 3.12 venv with `torch==2.3.1+cu118` so the V100 (sm_70) is supported.

## Pipeline

Three Slurm stages. Each stage produces input for the next, so chain them with `--dependency=afterok:<jobid>` if you want to run end-to-end.

### 1. Build pairs (CPU)

```
sbatch scripts/build_pairs.sh
```

Pulls transcripts from HuggingFace, fetches stock prices and EPS data via yfinance, strips boilerplate, and writes:

- `data/large/pairs_direction.jsonl`
- `data/large/pairs_eps_surprise.jsonl`

For just the surprise pairs with rate-limit-aware fetching (1.5s delay, retries, newest-first):

```
sbatch -M cardinal scripts/build_surprise.sh
```

### 2. Fine-tune (GPU)

Pick the script that matches the cluster you're scheduling on. Cardinal/Ascend run with `max_length=4096` and bf16. Pitzer (V100) runs with `max_length=2048` and fp16.

```
sbatch -M cardinal scripts/finetune_cardinal.sh direction
sbatch -M cardinal scripts/finetune_cardinal.sh surprise
sbatch -M ascend   scripts/finetune_ascend.sh   direction
sbatch -M pitzer   scripts/finetune_pitzer.sh   direction
```

Adapters land in `outputs/large/<task>/adapter/`.

### 3. Evaluate (GPU)

```
sbatch -M cardinal scripts/eval_cardinal.sh direction         # fine-tuned adapter
sbatch -M cardinal scripts/eval_cardinal.sh direction base    # zero-shot base
sbatch -M cardinal scripts/eval_cardinal.sh surprise
sbatch -M cardinal scripts/eval_cardinal.sh surprise base
```

Writes accuracy, balanced accuracy, AUC, and a 95% bootstrap CI to `outputs/large/<task>/results.json` (or `results_base.json` for the base model).

## Using a trained adapter without retraining

The trained LoRA adapters and tokenizer files are committed under `outputs/{small,large}/<task>/adapter/`. To run inference, load the base Qwen2.5-7B-Instruct model with the same 4-bit QLoRA config and attach the adapter via PEFT. See `load_model` in [src/models/evaluate.py](src/models/evaluate.py) for the exact loading code.

## Data sources

- Transcripts: `Bose345/sp500_earnings_transcripts` on HuggingFace (~33k S&P 500 calls).
- Stock prices: yfinance close-to-close % change over a configurable window.
- EPS surprise: `yfinance.Ticker(...).get_earnings_dates(...)` matched to the call by date.

yfinance fails for delisted tickers, recent IPOs, and corporate-action edge cases. The fetch helpers in [src/data/loader.py](src/data/loader.py) catch those and drop affected rows.

## Retrieval-Augmented Generation (RAG)

The project includes a RAG pipeline that augments the base and fine-tuned models with retrieved context from prior earnings calls.

We initially used sparse BM25 keyword retrieval as a simple baseline, then moved to dense retrieval using BAAI/bge-small-en-v1.5 embeddings indexed with FAISS. Dense retrieval finds semantically similar calls rather than relying on keyword overlap, which is better suited to the varied language of earnings transcripts.

At query time, retrieval combines a same-company heuristic (the most recent prior call from the same ticker) with nearest-neighbor search across the training split index. Retrieved passages are injected into the prompt alongside an optional set of few-shot labeled examples.

## Project Structure

```
src/retrieval/    RAG system: transcript embedding, FAISS indexing, retrieval logic, few-shot context building
src/models/       Fine-tuning and evaluation pipeline for Qwen2.5 models
scripts/slurm/    SLURM scripts used to run experiments on OSC Pitzer cluster
data/             Processed datasets — JSONL pairs for direction and EPS surprise tasks
outputs/          Model adapter weights and per-condition evaluation results
docs/             Final report and project writeups
```

## Running on OSC (Pitzer)

Experiments were run on the Ohio Supercomputer Center Pitzer cluster (V100 GPUs). Code and data were copied to OSC scratch space under `/fs/scratch/PAS3272/suryasuresh/finance_nlp_share`.

SLURM jobs were used for two stages:

- **Index building** (CPU jobs): build FAISS indices for each task before evaluation runs.
- **Evaluation** (GPU jobs): run inference across all experimental conditions.

Separate launcher scripts (`scripts/slurm/run_small.sh`, `run_large.sh`) submit the full condition matrix for the small and large datasets respectively. Some large-dataset evaluation runs did not fully complete within cluster time limits before the project deadline.
