"""
Evaluate a finetuned (or zero-shot base) model on the held-out chronological
test split of a JSONL pair file.

For each test example we run one forward pass and read the logits at the
final position.  We compare just the two target tokens (e.g. ' up' vs
' down'), softmax over that pair, and call the higher-probability one the
prediction.  Probability of the positive class is used for AUC.

Run:
    python -m src.models.evaluate \\
        --pairs    data/pairs_direction.jsonl \\
        --adapter  outputs/direction/adapter \\
        --out      outputs/direction/results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.finetune import load_pairs, head_tail_truncate, balance_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Per-task: positive class label (mapped to 1) and negative class label.
TASK_LABELS = {
    "direction": ("up",   "down"),
    "surprise":  ("beat", "miss"),
}


def detect_task(rows: list[dict]) -> str:
    """Look at the unique outputs to decide which task this file is for."""
    outs = {r["output"].strip().lower() for r in rows}
    if outs <= {"up", "down"}:    return "direction"
    if outs <= {"beat", "miss"}:  return "surprise"
    raise ValueError(f"Cannot detect task from outputs: {outs}")


def chrono_test_split(rows, train_ratio=0.9):
    """Same split as finetune.py — everything past the train cut is test."""
    n = len(rows)
    k = int(n * train_ratio)
    return rows[k:]


def first_subword_id(tokenizer, word: str) -> int:
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    assert ids, f"empty tokenization for {word!r}"
    return ids[0]


def load_model(base_model: str, adapter_dir):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # V100 (sm_70) has no bf16 support
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return model


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=0):
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(metric_fn(np.array(y_true)[idx], np.array(y_score)[idx]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@torch.no_grad()
def predict(model, tokenizer, rows, task, max_length):
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    probs_pos, y_true = [], []
    for i, row in enumerate(rows):
        prompt = row["input"] + "\n\nAnswer:"
        # Same head+tail truncation used during training — keeps opening
        # remarks AND the Q&A section.
        ids = head_tail_truncate(tokenizer, prompt, max_length)
        enc = {
            "input_ids":      torch.tensor([ids],            device=model.device),
            "attention_mask": torch.tensor([[1] * len(ids)], device=model.device),
        }
        logits = model(**enc).logits[0, -1]
        pair   = torch.tensor([logits[pos_id], logits[neg_id]])
        probs_pos.append(torch.softmax(pair, dim=0)[0].item())
        y_true.append(int(row["output"].strip().lower() == pos_word))
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(rows))

    return np.array(y_true), np.array(probs_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",      type=Path, required=True)
    ap.add_argument("--adapter",    type=str,  default=None,
                    help="Path to PEFT adapter dir. Omit for zero-shot base.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",        type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--balance-test", action="store_true",
                    help="Subsample majority class in test to match minority")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = load_pairs(args.pairs)
    task = detect_task(rows)
    test = chrono_test_split(rows, 0.9)
    logger.info("Task=%s  test=%d (before balancing)", task, len(test))
    if args.balance_test:
        test = balance_classes(test, seed=args.seed)
        from collections import Counter
        logger.info("Test balanced: %s  n=%d",
                    dict(Counter(r["output"] for r in test)), len(test))

    tok_src   = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model, args.adapter)
    y_true, y_prob = predict(model, tokenizer, test, task, args.max_length)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

    pos_word, neg_word = TASK_LABELS[task]
    results = {
        "task":              task,
        "pairs_file":        str(args.pairs),
        "adapter":           args.adapter,
        "balanced_test":     bool(args.balance_test),
        "n_test":            int(len(y_true)),
        "accuracy":          float(acc),
        "balanced_accuracy": float(bacc),
        "auc":               float(auc),
        "auc_ci_95":         [auc_lo, auc_hi],
        "class_balance":     {pos_word: int(y_true.sum()),
                              neg_word: int(len(y_true) - y_true.sum())},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
