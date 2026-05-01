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
