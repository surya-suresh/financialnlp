# Experiments and Extension Guide

A quick reference for what we ran, why, how evaluation works, and how to build on top of the trained adapters (RAG, prompting).

---

## 1. What we're predicting

Two separate binary classification tasks, both from the same earnings call transcripts:

| Task | Input | Label | Label source |
|---|---|---|---|
| **Direction** | Transcript + system prompt | `up` / `down` | Stock price change over 3 trading days (yfinance) |
| **Surprise** | Transcript + system prompt | `beat` / `miss` | Reported EPS vs. analyst consensus (yfinance) |

Each input is ~50,000 characters / 11,000 tokens. Model sees a truncated version (see §3).

---

## 2. What was run

### Run 1 (canonical results — currently in `outputs/`)

Single pass at each task with these settings:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Quantization | 4-bit NF4 + double-quant |
| Compute dtype | fp16 (V100 has no bf16) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `max_length` | 2048 tokens |
| Truncation | **Head + tail** (60% start + 40% end — keeps Q&A) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR schedule | Cosine with 3% warmup |
| Effective batch size | 8 (per-device 1 × grad-accum 8) |
| Optimizer | `paged_adamw_8bit` |
| Gradient checkpointing | Yes (needed for V100 16GB) |
| Best checkpoint | Selected by lowest val loss |
| Minority oversampling | Yes (train only, test stays natural) |
| Train/test split | 90 / 10 chronological (no shuffle) |
| GPU | V100-PCIE-16GB |
| Training time | ~47 min (direction) / ~50 min (surprise) |

### Run 2 (experimental, archived — `logs/archive/run2_balanced_failed/`)

Tried to enlarge surprise dataset + balance its test set:
- Bumped `--max-samples` from 2000 → 5000
- Added `--balance-test` flag that subsamples the majority class in test

Outcome: surprise pairs grew only from 827 → 840 (yfinance EPS coverage is the bottleneck, not source count). Balancing made the test *smaller* (83 → 34), so we reverted. The balancing flag is kept in the code for future use if more data becomes available.

---

## 3. Results (Run 1)

### Direction — no learnable signal

| | Fine-tuned | Base (zero-shot) |
|---|---|---|
| Test N | 197 | 197 |
| Accuracy | 0.594 | 0.569 |
| Balanced accuracy | 0.500 | 0.502 |
| AUC | 0.499 | 0.493 |
| AUC 95% CI | [0.42, 0.58] | [0.41, 0.57] |

**Interpretation:** both models sit at random chance. The fine-tuned accuracy of 0.594 matches the always-predict-"up" baseline exactly — the model just learned the majority class. Stock direction from transcripts alone is too noisy to learn from this dataset. Clean negative result.

### Surprise — strong signal

| | Fine-tuned | Base (zero-shot) |
|---|---|---|
| Test N | 83 | 83 |
| Class balance | 64 beat / 19 miss | 64 beat / 19 miss |
| Accuracy | 0.807 | 0.759 |
| Balanced accuracy | **0.708** | 0.566 |
| AUC | **0.813** | 0.637 |
| AUC 95% CI | **[0.69, 0.92]** | [0.49, 0.78] |

**Interpretation:** fine-tuning produces a real +0.18 AUC gain. The fine-tuned CI excludes random chance (0.50), meaning we can claim above-random performance with confidence. Executives signal beats/misses in how they talk about results; the model learned to pick up on that.

---

## 4. How evaluation works

### The split

Sort all pairs by date → take the last 10% as the test set. No random shuffling. This mimics deployment (train on past, predict future).

### The prediction

For each test row, we build the prompt:

```
<system prompt>

Transcript:
<head_tail_truncate(text, 2048)>

Answer:
```

Then run one forward pass through the model. We take the logits at the final position (i.e. the next-token distribution after "Answer:") and compare two specific token IDs: `' up'` vs `' down'` (or `' beat'` vs `' miss'`). Softmax over that pair gives a probability for the positive class.

We do **not** use `model.generate()` — it's unnecessary and slow when you only care about a single token choice. One forward pass is enough.

### Metrics computed

- **Accuracy** — `y_pred == y_true` at threshold 0.5
- **Balanced accuracy** — `(recall_positive + recall_negative) / 2`. Robust to class imbalance.
- **AUC (ROC)** — ranking quality, threshold-free
- **AUC 95% CI** — bootstrap with 1000 resamples
- **Class balance** — raw counts for sanity-checking

### Which metric to trust

- With imbalanced data: **balanced accuracy** and **AUC**, not plain accuracy. Plain accuracy can be inflated by just predicting the majority class.
- For ranking/deployment decisions: AUC (doesn't depend on the threshold you pick).
- For "is this better than random?": AUC CI excluding 0.50.
- For "is fine-tuned better than base?": the gap between point estimates + whether the CIs overlap.

---

## 5. Reusing the saved adapters

Every experiment saves a LoRA adapter (~40 MB) at `outputs/<task>/adapter/`. You can load it any time without retraining:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "outputs/surprise/adapter")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("outputs/surprise/adapter",
                                          trust_remote_code=True)
```

You still need a GPU node to run inference (model fits in ~6 GB VRAM with 4-bit), but loading is fast (~1 min) after the base model is cached.

**Critical:** the adapter expects the exact prompt format it was trained on. Keep the structure:

```
<system prompt>

Transcript:
<text>

Answer:
```

Changing the format (e.g., switching to a chat template) would undo your fine-tuning gains.

---

## 6. Adding RAG (retrieval-augmented generation)

**Goal:** inject extra context (past earnings calls, SEC filings, news) into the prompt so the model can reason over more than just the current call.

### Minimum viable RAG — 3 steps

#### Step 1: Build an index

Pick what you want to retrieve (e.g., prior transcripts from the same company). Embed each document with a sentence embedding model:

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

docs = [...]                               # list of strings
embeddings = encoder.encode(docs, normalize_embeddings=True)

import numpy as np, faiss                  # or just numpy for small corpora
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
```

#### Step 2: At inference, retrieve top-k

```python
query_emb = encoder.encode([current_transcript[:2000]],
                           normalize_embeddings=True)
_, idx = index.search(query_emb, k=3)      # top 3 docs
retrieved = [docs[i] for i in idx[0]]
```

Tip: embed only the *first* couple thousand characters of the query for speed — the full transcript would be overkill.

#### Step 3: Inject into the prompt

```python
context = "\n---\n".join(retrieved)
prompt = (
    f"{SYSTEM_PROMPT}\n\n"
    f"Relevant context from prior calls:\n{context}\n\n"
    f"Transcript:\n{current_transcript}\n\n"
    f"Answer:"
)
```

Then tokenize, truncate (use `head_tail_truncate` from `src/models/finetune.py`), and call the model exactly as in §4.

### Practical notes

- **Context length limit.** Qwen-7B supports 32k tokens natively, but your V100 fine-tuning config capped at 2048. For RAG inference you can push `max_length` up to ~4096 (forward-only needs less memory than training), but each retrieved doc you add costs you transcript tokens.
- **Where to inject.** Put retrieved context **before** "Transcript:", so the system prompt and final "Answer:" scaffolding stay exactly as the adapter saw them.
- **What to retrieve.** For surprise prediction: prior quarters' transcripts from the same company, analyst reports, sector news. For direction: same plus pre-call price momentum text ("AAPL is up 12% over the past 30 days").
- **Tiny index is fine.** If you only retrieve from ~1000 docs, numpy `argsort` on cosine similarity is faster to set up than FAISS.

---

## 7. Adding prompting methods

**Goal:** change what/how you ask the model, without retraining. All modifications happen to the prompt string fed to the model.

### Prompt structure

The adapter was trained to emit one answer token right after the literal `Answer:` marker. Any prompting changes need to keep this scaffolding intact:

```
<anything here>              ← system prompt, examples, reasoning, context
Transcript:
<text>

Answer:                      ← MUST be exactly this, at the very end
```

### Method 1: Few-shot (in-context examples)

Prepend 2-4 labeled examples before the current one:

```python
shots = [
    ("<transcript A>", "beat"),
    ("<transcript B>", "miss"),
]
few_shot = "\n\n".join(
    f"Transcript:\n{t}\n\nAnswer: {a}" for t, a in shots
)

prompt = (
    f"{SYSTEM_PROMPT}\n\n"
    f"Examples:\n{few_shot}\n\n"
    f"Transcript:\n{current}\n\n"
    f"Answer:"
)
```

Truncate each example to a few hundred tokens — don't burn your 2048 budget on shots.

### Method 2: Chain-of-thought

Ask for reasoning before the answer:

```python
prompt = (
    f"{SYSTEM_PROMPT}\n\n"
    f"Transcript:\n{current}\n\n"
    f"Think step by step about the management's tone, growth signals, "
    f"and margin commentary. Then give your answer.\n\n"
    f"Reasoning:"
)

reasoning = model.generate(prompt, max_new_tokens=200)
# then append "\n\nAnswer:" to the reasoning and run the single-token
# prediction exactly as in §4.
```

Two passes: one to generate reasoning (uses `.generate()`), a second to read the logits after "Answer:". CoT helps more on the surprise task where the model has real signal to reason over.

### Method 3: Self-consistency

Call the model `n` times with temperature > 0, take a majority vote:

```python
votes = []
for _ in range(5):
    prob_pos = single_forward(model, tokenizer, prompt, temperature=0.7)
    votes.append(1 if prob_pos > 0.5 else 0)
final = 1 if sum(votes) >= 3 else 0
```

Costs `n` × more inference. Modest gains on tasks with noisy signal.

### Method 4: Structured system prompt

Rewrite `_EPS_SYSTEM` / `_DIRECTION_SYSTEM` in `src/data/build_pairs.py` (or override at inference) with sharper instructions — e.g. a scoring rubric, a list of signals to look for, a definition of the label.

No retraining needed. Just a different system prompt at inference. Often cheaper than CoT and sometimes as effective.

### Evaluation

To measure any of these fairly, re-use the evaluation pipeline in `src/models/evaluate.py`. Build a small variant script that calls the adapter with your modified prompt, collects `probs_pos` and `y_true`, then feeds them to `accuracy_score`, `balanced_accuracy_score`, `roc_auc_score`, `bootstrap_ci`. Same metrics as §4, so results are directly comparable to the Run 1 numbers.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `libcudart.so.11.0: cannot open shared object file` | CUDA module not loaded | Add `module load cuda/11.8.0` in the job script |
| `ModuleNotFoundError: triton.ops` | Wrong triton version (3.x installed) | Pin `triton==2.3.1` (match torch 2.3.1) |
| `bf16/gpu` ValueError | Tried bf16 on V100 | Use `fp16=True` in `TrainingArguments` |
| `CUDA out of memory` on backward pass | `max_length` too high | Drop `max_length` from 3072 → 2048 |
| `pairs_eps_surprise.jsonl` is empty | `lxml` missing (yfinance needs it) | `pip install lxml` |
| Every EPS fetch returns None | Network or yfinance API change | Check loader.py line 98 column names |
| `use_synthetic_fallback` silently fires | CPU node couldn't import torch | `module load cuda/11.8.0` in build_pairs job too |

---

## 9. Where to look

| You want to... | File |
|---|---|
| Change dataset | `src/data/{loader,build_pairs}.py` |
| Change training | `src/models/finetune.py` |
| Change evaluation | `src/models/evaluate.py` |
| Submit a Slurm job | `scripts/run_*.sh` |
| See current metrics | `outputs/<task>/results*.json` |
| See past experiments | `logs/archive/` |
