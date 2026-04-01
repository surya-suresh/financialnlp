# Finance NLP Pipeline — Progress Report

**Project:** Does Financial Language Understanding Transfer to Market Prediction?
**Team:** Surya Suresh, Dheeraj Gosula, Nikhil Kumar

---

## Overview

End-to-end pipeline:

```
Earnings call transcripts
  → FinBERT sentiment extraction
  → Logistic regression classifier
  → Binary stock direction prediction (up / down)
```

**Baseline:** single sentiment score from the full transcript
**Improved:** separate scores for *prepared remarks* and *Q&A section*

---

## Repository Structure

```
src/
  data/
    loader.py        # transcript loading + yfinance price labels
    preprocessor.py  # text cleaning + section splitting
  features/
    sentiment.py     # FinBERT inference (chunked for long docs)
    extractor.py     # baseline vs improved feature sets
  models/
    classifier.py    # logistic regression pipeline
  utils/
    metrics.py       # accuracy / AUC + results table
  main.py            # orchestrates all steps

requirements.txt
setup.sh             # environment setup
run_job.sh           # Slurm job script for OSC Pitzer
package.sh           # creates project_osc.zip
README.md
```

---

## Running Locally

```bash
# 1. Set up environment
bash setup.sh
source venv/bin/activate

# 2. Run the full pipeline (real data, ~300 samples)
python src/main.py

# 3. Quick smoke test (no internet, no GPU needed)
python src/main.py --synthetic --max-samples 100
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-samples N` | 300 | Number of earnings calls to load |
| `--synthetic` | off | Use synthetic data (smoke test) |
| `--train-ratio F` | 0.8 | Fraction used for training |
| `--price-window N` | 3 | Trading days after call for price label |

---

## Running on OSC Pitzer

```bash
# 1. Transfer the zip to OSC
scp project_osc.zip <osc_username>@pitzer.osc.edu:~/

# 2. SSH into Pitzer and unzip
ssh <osc_username>@pitzer.osc.edu
unzip project_osc.zip
cd project

# 3. Set up environment (first time only)
bash setup.sh

# 4. Edit run_job.sh — replace <YOUR_OSC_PROJECT> with your allocation ID

# 5. Submit job
sbatch run_job.sh

# 6. Monitor job
squeue -u $USER
tail -f logs/output_<job_id>.log
```

---

## Expected Output

```
--------------------------------------------------
Model                         Accuracy     AUC-ROC
--------------------------------------------------
Baseline (full-transcript)      0.5800      0.6100
Improved (prepared + Q&A)       0.6100      0.6450
--------------------------------------------------
```

*Exact numbers will vary by dataset size and market conditions.*

---

## Where Logs Go

| File | Contents |
|------|----------|
| `logs/output_<job_id>.log` | stdout — step progress + final results table |
| `logs/error_<job_id>.log`  | stderr — warnings and errors |

---

## Data Sources

1. **Primary:** `Bose345/sp500_earnings_transcripts` (HuggingFace)
2. **Fallback:** `lamini/earnings-calls-qa` (HuggingFace)
3. **Smoke test:** synthetic data (built-in, no download needed)

Stock price labels fetched live via **yfinance**.

---

## Packaging for OSC

```bash
bash package.sh   # produces project_osc.zip
```
