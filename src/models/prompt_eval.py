import argparse
import json
import logging
import os
import random
import re
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


def chrono_train_split(rows: list[dict], train_ratio: float = 0.9) -> list[dict]:
    k = int(len(rows) * train_ratio)
    return rows[:k]


def extract_transcript(input_text: str) -> str:
    for marker in ("Transcript:\n", "Transcript:"):
        if marker in input_text:
            return input_text.split(marker, 1)[-1].strip()
    return input_text.strip()


def select_shots(train_rows: list[dict], n_shots: int,
                 pos_label: str, neg_label: str, seed: int = 42) -> list[dict]:
    if n_shots % 2 != 0:
        raise ValueError(f"n_shots must be even; got {n_shots}")

    rng = random.Random(seed)
    half = n_shots // 2

    def pick_from_class(rows: list[dict], k: int) -> list[dict]:
        n = len(rows)
        seg = max(n // k, 1)
        chosen = []
        for i in range(k):
            start = i * seg
            end = start + seg if i < k - 1 else n
            segment = rows[start:end]
            if segment:
                chosen.append(rng.choice(segment))
        return chosen

    pos_rows = [r for r in train_rows if r["output"].strip().lower() == pos_label]
    neg_rows = [r for r in train_rows if r["output"].strip().lower() == neg_label]

    pos_shots = pick_from_class(pos_rows, half)
    neg_shots = pick_from_class(neg_rows, half)

    # interleave pos and neg shots to avoid recency bias
    shots = []
    for p, n in zip(pos_shots, neg_shots):
        shots.append(p)
        shots.append(n)
    return shots


# financial keywords for scoring sentence relevance, sorted longest-first to avoid double-counting
_FINANCIAL_KEYWORDS = sorted([
    "earnings per share", "operating income", "net income", "per share",
    "eps", "revenue", "guidance", "outlook", "raised", "lowered",
    "below", "above", "margin", "growth", "decline", "consensus",
    "diluted", "forecast", "expect", "quarter", "fiscal", "sales",
    "profit", "loss",
], key=len, reverse=True)


def _score_sentence(sentence: str) -> int:
    remaining = sentence.lower()
    score = 0
    for kw in _FINANCIAL_KEYWORDS:
        if kw in remaining:
            score += 1
            remaining = remaining.replace(kw, " " * len(kw))
    return score


def extract_top_sentences(transcript: str, tokenizer, budget: int) -> str:
    cleaned = transcript.replace("\n", " ")
    raw_sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)\s+', cleaned)
                     if len(s.strip()) > 20]

    scored = []
    for s in raw_sentences:
        scored.append((_score_sentence(s), s))

    ranked = sorted(enumerate(scored), key=lambda x: x[1][0], reverse=True)

    chosen_indices, total_tokens = [], 0
    for orig_idx, (score, sentence) in ranked:
        if score == 0:
            break
        n_tok = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if total_tokens + n_tok > budget:
            continue
        chosen_indices.append(orig_idx)
        total_tokens += n_tok

    chosen_indices.sort()
    return ". ".join(scored[i][1] for i in chosen_indices)


def build_shot_excerpts(shots: list[dict], tokenizer, excerpt_tokens: int) -> list[str]:
    excerpts = []
    for row in shots:
        raw     = extract_transcript(row["input"])
        excerpt = extract_top_sentences(raw, tokenizer, excerpt_tokens)
        excerpts.append(excerpt)
    return excerpts


def build_prompt(system: str, shots: list[dict], shot_excerpts: list[str],
                 shot_context_blocks: list[str],
                 tokenizer, test_input: str, max_length: int,
                 context_block: str = "") -> str:
    header = (
        f"{system}\n\n"
        f"Below are {len(shots)} examples with known outcomes.\n"
    )

    ex_block = ""
    for i, (row, excerpt, sctx) in enumerate(zip(shots, shot_excerpts, shot_context_blocks), 1):
        ctx_part = f"{sctx}\n" if sctx else ""
        ex_block += (
            f"\nExample {i}:\n"
            f"{ctx_part}"
            f"Transcript: {excerpt}\n"
            f"Answer: {row['output']}\n"
        )

    ctx_section = f"\n{context_block}\n" if context_block else ""
    test_header = f"\nNow analyze the following:{ctx_section}\nTranscript: "
    suffix      = "\nAnswer:"

    framework     = header + ex_block + test_header + suffix
    framework_len = len(tokenizer(framework, add_special_tokens=False)["input_ids"])
    test_budget   = max(max_length - framework_len, 64)

    raw_test = extract_transcript(test_input)

    # iteratively shrink budget if tokenization drift causes an overrun
    for _ in range(20):
        test_text   = extract_top_sentences(raw_test, tokenizer, test_budget)
        full_prompt = header + ex_block + test_header + test_text + suffix
        if len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"]) <= max_length:
            break
        test_budget = max(test_budget - 16, 64)

    return full_prompt


@torch.no_grad()
def predict(model, tokenizer, prompts: list[str], labels: list[int],
            task: str, max_length: int):
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    probs_pos, y_true = [], []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            raise RuntimeError(f"Prompt {i} exceeds max_length={max_length}")
        tail_text = tokenizer.decode(ids[-10:], skip_special_tokens=True)
        if not tail_text.rstrip().endswith("Answer:"):
            raise RuntimeError(f"Prompt {i} does not end with 'Answer:': {tail_text!r}")
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


def main():
    ap = argparse.ArgumentParser(description="4-shot few-shot prompting evaluation")
    ap.add_argument("--pairs",          type=Path, required=True)
    ap.add_argument("--adapter",        type=str,  default=None)
    ap.add_argument("--base-model",     default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",            type=Path, required=True)
    ap.add_argument("--n-shots",        type=int,  default=4)
    ap.add_argument("--excerpt-tokens", type=int,  default=150)
    ap.add_argument("--max-length",     type=int,  default=3072)
    ap.add_argument("--min-eps-margin", type=float, default=0.0)
    ap.add_argument("--balance-test",   action="store_true")
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--context",        action="store_true")
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
    logger.info("%d-shot examples: %s", len(shots),
                [(s["ticker"], s["date"], s["output"]) for s in shots])

    tok_src   = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system = TASK_SYSTEMS[task]

    logger.info("Pre-building shot excerpts ...")
    shot_excerpts = build_shot_excerpts(shots, tokenizer, args.excerpt_tokens)

    context_map: dict = {}
    shot_context_blocks: list[str] = [""] * len(shots)
    if args.context:
        logger.info("Fetching context blocks via yfinance ...")
        context_map = fetch_context_blocks_batch(shots + test)
        shot_context_blocks = [
            context_map.get((s["ticker"], s["date"][:10]), "") for s in shots
        ]

    logger.info("Building %d prompts ...", len(test))
    prompts, labels = [], []
    for row in test:
        ctx_block = context_map.get((row["ticker"], row["date"][:10]), "")
        prompt = build_prompt(
            system, shots, shot_excerpts, shot_context_blocks, tokenizer,
            row["input"], args.max_length,
            context_block=ctx_block,
        )
        prompts.append(prompt)
        labels.append(int(row["output"].strip().lower() == pos_word))

    logger.info("Sample prompt:\n%s", prompts[0][:1000])

    model = load_model(args.base_model, args.adapter)
    logger.info("Running inference on %d test examples ...", len(prompts))
    y_true, y_prob = predict(model, tokenizer, prompts, labels, task, args.max_length)
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

    results = {
        "task":              task,
        "pairs_file":        str(args.pairs),
        "adapter":           args.adapter,
        "n_shots":           args.n_shots,
        "seed":              args.seed,
        "max_length":        args.max_length,
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
    logger.info("Saved -> %s", args.out)
    logger.info("Results:\n%s",
                json.dumps({k: v for k, v in results.items() if k != "shots"}, indent=2))


if __name__ == "__main__":
    main()
