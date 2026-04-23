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


def extract_transcript(input_text: str) -> str:
    """Pull just the transcript body from the stored prompt string."""
    for marker in ("Transcript:\n", "Transcript:"):
        if marker in input_text:
            return input_text.split(marker, 1)[-1].strip()
    return input_text.strip()


def prompt_prefix(input_text: str) -> str:
    """Return everything before the transcript marker."""
    for marker in ("Transcript:\n", "Transcript:"):
        if marker in input_text:
            return input_text.split(marker, 1)[0].strip()
    return ""


def eps_margin(row: dict) -> float | None:
    """Absolute reported-vs-consensus EPS gap, when available."""
    try:
        reported = row.get("reported_eps")
        consensus = row.get("consensus_eps")
        if reported is None or consensus is None:
            return None
        return abs(float(reported) - float(consensus))
    except (TypeError, ValueError):
        return None


def filter_by_eps_margin(rows: list[dict], min_margin: float) -> list[dict]:
    """Drop EPS surprise examples whose reported/consensus gap is too small."""
    if min_margin <= 0:
        return rows
    kept = [r for r in rows if (eps_margin(r) is not None and eps_margin(r) >= min_margin)]
    logger.info(
        "EPS margin filter %.4f: kept %d / %d rows",
        min_margin,
        len(kept),
        len(rows),
    )
    return kept


def split_transcript_sections(transcript: str) -> tuple[str, str]:
    """Best-effort prepared remarks / Q&A split for earnings-call transcripts."""
    lower = transcript.lower()
    qa_markers = [
        "question-and-answer session",
        "question and answer session",
        "questions-and-answers session",
        "questions and answers session",
        "q&a session",
        "q & a session",
    ]
    positions = [lower.find(marker) for marker in qa_markers if lower.find(marker) >= 0]
    if not positions:
        # Intro boilerplate can contain "[Operator Instructions]", so only use
        # it as a fallback after enough of the call has elapsed.
        min_pos = min(max(len(transcript) // 5, 1000), 5000)
        op_pos = lower.find("[operator instructions]", min_pos)
        if op_pos >= 0:
            positions.append(op_pos)
    if not positions:
        return transcript, transcript
    split_at = min(positions)
    return transcript[:split_at].strip(), transcript[split_at:].strip()


def select_transcript_view(transcript: str, view: str) -> tuple[str, float | None]:
    """
    Return the requested transcript view plus an optional head/tail ratio.
    A None ratio means plain front truncation after the view is selected.
    """
    prepared, qa = split_transcript_sections(transcript)
    if view == "head_tail":
        return transcript, 0.6
    if view == "tail_heavy":
        return transcript, 0.3
    if view == "front":
        return transcript, None
    if view == "prepared":
        return prepared or transcript, None
    if view == "qa":
        return qa or transcript, None
    raise ValueError(f"Unknown transcript view: {view}")


def build_view_prompt_ids(tokenizer, row: dict, view: str, max_length: int) -> list[int]:
    """Build token ids for one transcript view while preserving Answer: at the end."""
    prefix = prompt_prefix(row["input"])
    transcript = extract_transcript(row["input"])
    view_text, head_ratio = select_transcript_view(transcript, view)

    before = f"{prefix}\n\nTranscript:\n" if prefix else "Transcript:\n"
    suffix = "\n\nAnswer:"
    framework_len = len(tokenizer(before + suffix, add_special_tokens=False)["input_ids"])
    transcript_budget = max(max_length - framework_len, 64)

    if head_ratio is None:
        view_ids = tokenizer(view_text, add_special_tokens=False)["input_ids"][:transcript_budget]
    else:
        view_ids = head_tail_truncate(tokenizer, view_text, transcript_budget, head_ratio=head_ratio)

    before_ids = tokenizer(before, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    return before_ids + view_ids + suffix_ids


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
def predict(model, tokenizer, rows, task, max_length, views: list[str]):
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    view_scores = {view: [] for view in views}
    y_true = []
    for i, row in enumerate(rows):
        for view in views:
            ids = build_view_prompt_ids(tokenizer, row, view, max_length)
            enc = {
                "input_ids":      torch.tensor([ids],            device=model.device),
                "attention_mask": torch.tensor([[1] * len(ids)], device=model.device),
            }
            logits = model(**enc).logits[0, -1]
            view_scores[view].append((logits[pos_id] - logits[neg_id]).item())
        y_true.append(int(row["output"].strip().lower() == pos_word))
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(rows))

    y_true_arr = np.array(y_true)
    per_view_prob = {
        view: 1.0 / (1.0 + np.exp(-np.array(scores)))
        for view, scores in view_scores.items()
    }
    avg_score = np.mean(np.stack([np.array(view_scores[v]) for v in views], axis=0), axis=0)
    ensemble_prob = 1.0 / (1.0 + np.exp(-avg_score))
    return y_true_arr, ensemble_prob, per_view_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",      type=Path, required=True)
    ap.add_argument("--adapter",    type=str,  default=None,
                    help="Path to PEFT adapter dir. Omit for zero-shot base.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",        type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--views",      default="head_tail",
                    help=("Comma-separated transcript views. Options: head_tail,"
                          "tail_heavy,front,prepared,qa. Multiple views are"
                          " ensembled by averaging logit margins."))
    ap.add_argument("--min-eps-margin", type=float, default=0.0,
                    help=("For EPS surprise, drop rows with abs(reported_eps - "
                          "consensus_eps) below this value before splitting."))
    ap.add_argument("--balance-test", action="store_true",
                    help="Subsample majority class in test to match minority")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = load_pairs(args.pairs)
    rows = filter_by_eps_margin(rows, args.min_eps_margin)
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
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    logger.info("Transcript views: %s", views)
    y_true, y_prob, per_view_prob = predict(
        model, tokenizer, test, task, args.max_length, views,
    )
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

    pos_word, neg_word = TASK_LABELS[task]
    per_view_metrics = {}
    for view, probs in per_view_prob.items():
        try:
            view_auc = roc_auc_score(y_true, probs)
        except ValueError:
            view_auc = float("nan")
        view_pred = (probs >= 0.5).astype(int)
        per_view_metrics[view] = {
            "accuracy": float(accuracy_score(y_true, view_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, view_pred)),
            "auc": float(view_auc),
        }

    results = {
        "task":              task,
        "pairs_file":        str(args.pairs),
        "adapter":           args.adapter,
        "balanced_test":     bool(args.balance_test),
        "min_eps_margin":    float(args.min_eps_margin),
        "views":             views,
        "n_test":            int(len(y_true)),
        "accuracy":          float(acc),
        "balanced_accuracy": float(bacc),
        "auc":               float(auc),
        "auc_ci_95":         [auc_lo, auc_hi],
        "per_view":          per_view_metrics,
        "class_balance":     {pos_word: int(y_true.sum()),
                              neg_word: int(len(y_true) - y_true.sum())},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
