"""
prompt_eval.py — 3-shot few-shot prompting evaluation (base or finetuned model).

Each few-shot demonstration shows a transcript excerpt and its known outcome.
The model predicts via logit comparison at the "Answer:" position —
identical to evaluate.py so results are directly comparable.

Run:
    # base model, 3-shot, direction
    python -m src.models.prompt_eval \\
        --pairs data/pairs_direction.jsonl \\
        --out   outputs/direction/results_prompt_base.json

    # finetuned model, 3-shot, direction
    python -m src.models.prompt_eval \\
        --pairs   data/pairs_direction.jsonl \\
        --adapter outputs/direction/adapter \\
        --out     outputs/direction/results_prompt_ft.json

    # base model, 3-shot, surprise
    python -m src.models.prompt_eval \\
        --pairs data/pairs_eps_surprise.jsonl \\
        --out   outputs/surprise/results_prompt_base.json

    # finetuned model, 3-shot, surprise
    python -m src.models.prompt_eval \\
        --pairs   data/pairs_eps_surprise.jsonl \\
        --adapter outputs/surprise/adapter \\
        --out     outputs/surprise/results_prompt_ft.json
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.finetune import load_pairs, balance_classes, head_tail_truncate
from models.evaluate import (
    TASK_LABELS, detect_task, chrono_test_split,
    first_subword_id, load_model, bootstrap_ci, filter_by_eps_margin,
)
from data.loader import fetch_context_blocks_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_DIRECTION_SYSTEM = (
    "You are a financial analyst predicting stock price direction from earnings "
    "call transcripts. You will be shown example transcripts with known outcomes, "
    "then asked to predict a new one. "
    "Respond with exactly one word: 'up' or 'down'."
)

_EPS_SYSTEM = (
    "You are a financial analyst predicting earnings surprises from earnings "
    "call transcripts. You will be shown example transcripts with known outcomes, "
    "then asked to predict a new one. "
    "Respond with exactly one word: 'beat' or 'miss'."
)

TASK_SYSTEMS = {
    "direction": _DIRECTION_SYSTEM,
    "surprise":  _EPS_SYSTEM,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def chrono_train_split(rows: list[dict], train_ratio: float = 0.9) -> list[dict]:
    k = int(len(rows) * train_ratio)
    return rows[:k]


def extract_transcript(input_text: str) -> str:
    """Pull just the transcript body from the stored prompt string."""
    for marker in ("Transcript:\n", "Transcript:"):
        if marker in input_text:
            return input_text.split(marker, 1)[-1].strip()
    return input_text.strip()


def select_shots(train_rows: list[dict], n_shots: int,
                 pos_label: str, neg_label: str, seed: int = 42) -> list[dict]:
    """
    Pick n_shots spread across evenly-spaced temporal segments of the training
    set, ensuring at least 1 positive and 1 negative example.
    Temporal diversity prevents all shots coming from the same market period.
    """
    rng = random.Random(seed)
    n = len(train_rows)

    if n == 0:
        return []

    # Pick one example from each of n_shots equal temporal segments
    segment_size = max(n // n_shots, 1)
    candidates = []
    for i in range(n_shots):
        start = i * segment_size
        end = start + segment_size if i < n_shots - 1 else n
        segment = train_rows[start:end]
        if segment:
            candidates.append(rng.choice(segment))

    # Guarantee at least 1 of each class — replace last candidate if needed
    labels_present = {c["output"].strip().lower() for c in candidates}
    for missing_label in [pos_label, neg_label]:
        if missing_label not in labels_present:
            pool = [r for r in train_rows if r["output"].strip().lower() == missing_label]
            if pool:
                candidates[-1] = rng.choice(pool)
                labels_present = {c["output"].strip().lower() for c in candidates}

    rng.shuffle(candidates)
    return candidates[:n_shots]


def build_prompt(system: str, shots: list[dict], tokenizer,
                 excerpt_tokens: int, test_input: str,
                 max_length: int,
                 context_block: str = "") -> str:
    """
    Assemble the full k-shot prompt ending with "Answer:".

    Layout (with optional context_block):
        {system}

        Below are {k} examples with known outcomes.

        Example 1:
        Transcript: {short excerpt}
        Answer: {label}

        ...

        Now analyze the following:
        {context_block}        ← optional non-transcript features
        Transcript: {test transcript, head+tail truncated to fit budget}
        Answer:
    """
    header = (
        f"{system}\n\n"
        f"Below are {len(shots)} examples with known outcomes.\n"
    )

    ex_block = ""
    for i, row in enumerate(shots, 1):
        raw     = extract_transcript(row["input"])
        ids     = tokenizer(raw, add_special_tokens=False)["input_ids"][:excerpt_tokens]
        excerpt = tokenizer.decode(ids, skip_special_tokens=True).strip()
        ex_block += (
            f"\nExample {i}:\n"
            f"Transcript: {excerpt}\n"
            f"Answer: {row['output']}\n"
        )

    # Context block (non-transcript numerical features) injected before transcript
    ctx_section = f"\n{context_block}\n" if context_block else ""
    test_header = f"\nNow analyze the following:{ctx_section}\nTranscript: "
    suffix      = "\nAnswer:"

    # Compute remaining token budget for the test transcript
    framework     = header + ex_block + test_header + suffix
    framework_len = len(tokenizer(framework, add_special_tokens=False)["input_ids"])
    test_budget   = max(max_length - framework_len - 4, 64)

    raw_test  = extract_transcript(test_input)
    test_ids  = head_tail_truncate(tokenizer, raw_test, test_budget)
    test_text = tokenizer.decode(test_ids, skip_special_tokens=True).strip()

    return header + ex_block + test_header + test_text + suffix


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tokenizer, prompts: list[str], labels: list[int],
            task: str, max_length: int):
    """
    Read the logit at the final "Answer:" position and compare the two
    target tokens — same method as evaluate.py for direct comparability.
    """
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    probs_pos, y_true = [], []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            ids = ids[:max_length]
        enc = {
            "input_ids":      torch.tensor([ids],            device=model.device),
            "attention_mask": torch.tensor([[1] * len(ids)], device=model.device),
        }
        logits = model(**enc).logits[0, -1]
        pair   = torch.tensor([logits[pos_id], logits[neg_id]])
        probs_pos.append(torch.softmax(pair, dim=0)[0].item())
        y_true.append(label)
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(prompts))

    return np.array(y_true), np.array(probs_pos)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="3-shot few-shot prompting evaluation")
    ap.add_argument("--pairs",          type=Path, required=True,
                    help="JSONL pairs file (e.g. data/pairs_direction.jsonl)")
    ap.add_argument("--adapter",        type=str,  default=None,
                    help="PEFT adapter dir. Omit for base model.")
    ap.add_argument("--base-model",     default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",            type=Path, required=True,
                    help="Output JSON file for results")
    ap.add_argument("--n-shots",        type=int,  default=3)
    ap.add_argument("--excerpt-tokens", type=int,  default=200,
                    help="Token budget per demonstration excerpt (default 200)")
    ap.add_argument("--max-length",     type=int,  default=2048)
    ap.add_argument("--min-eps-margin", type=float, default=0.0,
                    help=("For EPS surprise, drop rows with abs(reported_eps - "
                          "consensus_eps) below this value before splitting."))
    ap.add_argument("--balance-test",   action="store_true",
                    help="Subsample majority class in test to match minority")
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--context",        action="store_true",
                    help=("Prepend a non-transcript data block to each test prompt: "
                          "prior 4Q EPS, analyst consensus, sector, YTD performance. "
                          "Fetched via yfinance at eval time. No reported EPS included."))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = load_pairs(args.pairs)
    rows = filter_by_eps_margin(rows, args.min_eps_margin)
    task = detect_task(rows)
    logger.info("Task: %s  |  total pairs: %d", task, len(rows))

    train = chrono_train_split(rows, 0.9)
    test  = chrono_test_split(rows, 0.9)
    logger.info("Train=%d  Test=%d (before balancing)", len(train), len(test))

    if args.balance_test:
        test = balance_classes(test, seed=args.seed)
        logger.info("Test balanced to %d examples", len(test))

    pos_word, neg_word = TASK_LABELS[task]
    shots = select_shots(train, args.n_shots, pos_word, neg_word, seed=args.seed)
    logger.info(
        "%d-shot examples selected: %s",
        len(shots),
        [(s["ticker"], s["date"], s["output"]) for s in shots],
    )

    tok_src   = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system = TASK_SYSTEMS[task]

    # Pre-fetch context blocks (one batch of yfinance calls) if requested
    context_map: dict = {}
    if args.context:
        logger.info("Fetching non-transcript context blocks via yfinance …")
        context_map = fetch_context_blocks_batch(test)
        n_filled = sum(1 for v in context_map.values() if v)
        logger.info("Context blocks: %d / %d non-empty", n_filled, len(test))

    logger.info("Building %d prompts …", len(test))
    prompts, labels = [], []
    for row in test:
        ctx_block = context_map.get((row["ticker"], row["date"][:10]), "")
        prompt = build_prompt(
            system, shots, tokenizer,
            args.excerpt_tokens, row["input"], args.max_length,
            context_block=ctx_block,
        )
        prompts.append(prompt)
        labels.append(int(row["output"].strip().lower() == pos_word))

    logger.info("─── Sample prompt (first 1000 chars) ───\n%s\n───", prompts[0][:1000])

    model = load_model(args.base_model, args.adapter)
    logger.info("Running inference on %d test examples …", len(prompts))
    y_true, y_prob = predict(model, tokenizer, prompts, labels, task, args.max_length)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

    results = {
        "task":              task,
        "pairs_file":        str(args.pairs),
        "adapter":           args.adapter,
        "n_shots":           args.n_shots,
        "seed":              args.seed,
        "cot":               False,
        "context":           bool(args.context),
        "excerpt_tokens":    args.excerpt_tokens,
        "balanced_test":     bool(args.balance_test),
        "min_eps_margin":    float(args.min_eps_margin),
        "n_test":            int(len(y_true)),
        "accuracy":          float(acc),
        "balanced_accuracy": float(bacc),
        "auc":               float(auc),
        "auc_ci_95":         [auc_lo, auc_hi],
        "class_balance":     {pos_word: int(y_true.sum()),
                              neg_word: int(len(y_true) - y_true.sum())},
        "shots":             [{"ticker": s["ticker"], "date": s["date"],
                               "output": s["output"]} for s in shots],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Saved → %s", args.out)
    logger.info(
        "Results:\n%s",
        json.dumps({k: v for k, v in results.items() if k != "shots"}, indent=2),
    )


if __name__ == "__main__":
    main()
