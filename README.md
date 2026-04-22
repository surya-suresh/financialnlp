# Finance NLP — Earnings Call Prediction

**Project:** Does Financial Language Understanding Transfer to Market Prediction?
**Team:** Surya Suresh, Dheeraj Gosula, Nikhil Kumar

Fine-tune Qwen2.5-7B-Instruct (QLoRA) on S&P 500 earnings call transcripts to predict:

1. **Direction** — will the stock go up or down over 3 trading days?
2. **Surprise** — will EPS beat or miss analyst consensus?

Also includes an older FinBERT + logistic-regression baseline (`src/main.py`, `scripts/run_job.sh`).

---

## Repository layout

```
financialnlp/
├── README.md
├── requirements.txt
├── setup.sh                         # environment setup
├── package.sh                       # zip the project for transfer
│
├── docs/                            # proposal + LaTeX style
│   ├── proposal.pdf
│   ├── proposal.tex
│   └── acl.sty
│
├── src/
│   ├── data/
│   │   ├── loader.py                # transcript loading + yfinance price/EPS labels
│   │   ├── preprocessor.py          # text cleaning + section splitting (FinBERT pipeline)
│   │   └── build_pairs.py           # writes pairs_direction.jsonl + pairs_eps_surprise.jsonl
│   ├── features/                    # FinBERT baseline features
│   │   ├── sentiment.py
│   │   └── extractor.py
│   ├── models/
│   │   ├── classifier.py            # (baseline) logistic regression on FinBERT features
│   │   ├── finetune.py              # QLoRA finetune Qwen2.5-7B on one task
│   │   └── evaluate.py              # evaluate adapter or base on held-out test
│   ├── utils/metrics.py
│   └── main.py                      # (baseline) FinBERT → LogReg end-to-end
│
├── scripts/                         # Slurm job scripts + utilities
│   ├── run_build_pairs.sh           # fetch labels, produce JSONL pairs (CPU)
│   ├── run_finetune.sh              # QLoRA finetune (GPU)
│   ├── run_eval.sh                  # evaluate adapter or zero-shot base (GPU)
│   ├── run_job.sh                   # (baseline) FinBERT pipeline (GPU)
│   └── smoke_test.py                # diagnostic: load model + one forward pass
│
├── data/
│   ├── pairs_direction.jsonl        # current direction pairs
│   ├── pairs_eps_surprise.jsonl     # current surprise pairs
│   └── archive/                     # snapshots of earlier data pulls
│
├── outputs/
│   ├── direction/                   # results.json, results_base.json, adapter/
│   ├── surprise/                    # results.json, results_base.json, adapter/
│   └── archive/                     # snapshots of earlier runs
│
└── logs/
    ├── <build_pairs|finetune|eval>_<jobid>.(out|err)
    └── archive/                     # logs from earlier runs
```

---

## Setup (first time only)

```bash
bash setup.sh
source venv/bin/activate
```

This builds a Python 3.9 venv, installs `torch==2.3.1+cu118` (V100-compatible), and the rest of the deps. Takes ~5 minutes.

---

## The main pipeline (QLoRA on earnings calls)

The three-stage pipeline is run as Slurm jobs on OSC Pitzer. Each stage depends on the previous one, so use `--dependency=afterok:<job_id>` when chaining.

### Stage 1 — Build pairs (CPU, ~30–90 min)

Fetches S&P 500 transcripts from HuggingFace, fetches stock prices + EPS data from yfinance, and writes two JSONL files with one `(input, output)` pair per row.

```bash
sbatch scripts/run_build_pairs.sh
```

Outputs:
- `data/pairs_direction.jsonl` — input=prompt+transcript, output="up" or "down"
- `data/pairs_eps_surprise.jsonl` — input=prompt+transcript, output="beat" or "miss"

### Stage 2 — Fine-tune (GPU, V100, ~1–5h per task)

```bash
sbatch scripts/run_finetune.sh direction
sbatch scripts/run_finetune.sh surprise
```

Each job QLoRA-finetunes Qwen2.5-7B-Instruct (4-bit NF4 + LoRA r=16) on 90% of the pairs (chronological split). Saves the adapter + tokenizer to `outputs/<task>/adapter/`.

### Stage 3 — Evaluate (GPU, V100, ~1–5 min per eval)

```bash
sbatch scripts/run_eval.sh direction              # finetuned adapter
sbatch scripts/run_eval.sh direction base         # zero-shot base for comparison
sbatch scripts/run_eval.sh surprise
sbatch scripts/run_eval.sh surprise base
```

Writes JSON metrics (accuracy, balanced accuracy, AUC + 95% CI, class balance) to `outputs/<task>/results.json` (adapter) and `outputs/<task>/results_base.json` (base).

### Chaining all stages in one go

```bash
BP=$(sbatch --parsable scripts/run_build_pairs.sh)
FT_D=$(sbatch --parsable --dependency=afterok:$BP scripts/run_finetune.sh direction)
FT_S=$(sbatch --parsable --dependency=afterok:$BP scripts/run_finetune.sh surprise)
sbatch --dependency=afterok:$FT_D scripts/run_eval.sh direction
sbatch --dependency=afterok:$FT_S scripts/run_eval.sh surprise
sbatch --dependency=afterok:$BP  scripts/run_eval.sh direction base
sbatch --dependency=afterok:$BP  scripts/run_eval.sh surprise base
```

### Monitoring

```bash
squeue -u $USER --start                   # queue + ETA
tail -f logs/finetune_<jobid>.err         # live training output
```

---

## Environment requirements on OSC Pitzer

The GPU job scripts assume V100-compatible modules. They load:
- `python/3.12` — activates the module system's Python (venv overrides it)
- `cuda/11.8.0` — `libcudart.so.11.0` for torch 2.3.1+cu118
- `cudnn/8.7.0.84-11.8` — `libcudnn.so.8`
- Adds `$CUDA_HOME/extras/CUPTI/lib64` to `LD_LIBRARY_PATH` for `libcupti.so.11.8`

Do not change these without also changing `setup.sh` and verifying a smoke test passes.

---

## Data sources

1. **Transcripts:** `Bose345/sp500_earnings_transcripts` on HuggingFace (33k calls, mostly 2024-2025).
2. **Stock prices:** `yfinance` — live fetch, percent change from close-before-call to close-N-trading-days-after.
3. **EPS surprise:** `yfinance.Ticker(...).get_earnings_dates(...)` — actual vs. analyst consensus EPS.

`yfinance` is noisy: it fails for delisted tickers, recent IPOs, and tickers in unusual corporate states (mergers, splits). `_fetch_price_change()` and `_fetch_eps_surprise()` catch these errors and drop affected rows.

---

## Current headline results (run 1 — N=197 direction / N=83 surprise)

| Task | Model | Accuracy | Balanced acc | AUC | 95% CI |
|---|---|---|---|---|---|
| Direction | Fine-tuned | 0.594 | 0.500 | 0.499 | [0.42, 0.58] |
| Direction | Base | 0.569 | 0.502 | 0.493 | [0.41, 0.57] |
| Surprise | **Fine-tuned** | **0.807** | **0.708** | **0.813** | **[0.69, 0.92]** |
| Surprise | Base | 0.759 | 0.566 | 0.637 | [0.49, 0.78] |

- **Direction**: fine-tuning does not help. Transcripts alone don't carry enough signal for 3-day stock direction.
- **Surprise**: fine-tuning works. AUC 0.81 with CI excluding random (0.50).

See `outputs/<task>/results*.json` for full metrics and `outputs/archive/` for snapshots of prior runs.
