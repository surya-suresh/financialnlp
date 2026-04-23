"""
prompt_eval.py — 3-shot chain-of-thought prompting evaluation.

Each few-shot demonstration includes a Reasoning step before the Answer,
showing the model what analytical pattern to follow.  For the test example
the model generates its own reasoning, then we read the logit at the
appended "Answer:" position — combining CoT with deterministic evaluation.

Run:
    # base model, 3-shot CoT, direction
    python -m src.models.prompt_eval \\
        --pairs data/pairs_direction.jsonl \\
        --out   outputs/direction/results_prompt_base.json

    # finetuned model, 3-shot CoT, direction
    python -m src.models.prompt_eval \\
        --pairs   data/pairs_direction.jsonl \\
        --adapter outputs/direction/adapter \\
        --out     outputs/direction/results_prompt_ft.json

    # base model, 3-shot CoT, surprise
    python -m src.models.prompt_eval \\
        --pairs data/pairs_eps_surprise.jsonl \\
        --out   outputs/surprise/results_prompt_base.json

    # finetuned model, 3-shot CoT, surprise
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
    first_subword_id, load_model, bootstrap_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_DIRECTION_SYSTEM = (
    "You are a financial analyst predicting stock price direction from earnings "
    "call transcripts. You will be shown example transcripts, each with a step-by-step "
    "reasoning and a known outcome. Then you will analyze a new transcript using the "
    "same reasoning process before giving your prediction. "
    "Your final answer must be exactly one word: 'up' or 'down'."
)

_EPS_SYSTEM = (
    "You are a financial analyst predicting earnings surprises from earnings "
    "call transcripts. You will be shown example transcripts, each with a step-by-step "
    "reasoning and a known outcome. Then you will analyze a new transcript using the "
    "same reasoning process before giving your prediction. "
    "Your final answer must be exactly one word: 'beat' or 'miss'."
)

TASK_SYSTEMS = {
    "direction": _DIRECTION_SYSTEM,
    "surprise":  _EPS_SYSTEM,
}

# ── CoT reasoning templates for few-shot demonstrations ───────────────────────
# These show the model what analytical patterns to look for and how to
# articulate reasoning before committing to an answer.

_SHOT_REASONING = {
    "direction": {
        "up": (
            "Management expressed confidence in the demand environment and raised or "
            "reaffirmed full-year guidance. Gross margins held firm or expanded, and "
            "forward commentary was constructive. Analyst questions were met with "
            "direct, positive responses rather than hedging. These signals collectively "
            "indicate investor sentiment is likely to improve in the near term."
        ),
        "down": (
            "Management acknowledged headwinds — softening demand, margin compression, "
            "or a guidance reduction — and the prepared remarks contained cautious "
            "language around the macro environment. Analysts pressed on execution "
            "concerns and received hedged or defensive answers. These signals suggest "
            "the market reaction is likely to be negative over the next three trading days."
        ),
    },
    "surprise": {
        "beat": (
            "The reported EPS and revenue figures appear to exceed what the market "
            "expected. Management's tone was upbeat and they referenced outperformance "
            "on key metrics relative to prior guidance. The Q&A section lacked the "
            "defensive posture typical of a miss, with analysts focusing on upside "
            "drivers rather than shortfalls. This points to a positive earnings surprise."
        ),
        "miss": (
            "The reported figures fell short of prior guidance and analyst estimates. "
            "Management was defensive on key metrics and spent time explaining "
            "execution challenges or external headwinds. Analysts focused on the "
            "shortfall rather than on growth opportunities. This pattern is consistent "
            "with an earnings miss relative to consensus."
        ),
    },
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
    Pick n_shots from the chronologically most-recent training examples,
    ensuring at least 1 positive and 1 negative example.
    """
    rng = random.Random(seed)
    pos = [r for r in train_rows if r["output"].strip().lower() == pos_label]
    neg = [r for r in train_rows if r["output"].strip().lower() == neg_label]

    if pos and neg:
        n_pos = max(1, n_shots // 2)
        n_neg = n_shots - n_pos
        shots = pos[-n_pos:] + neg[-n_neg:]
    else:
        shots = train_rows[-n_shots:]

    rng.shuffle(shots)
    return shots[:n_shots]


def build_prompt(system: str, shots: list[dict], task: str, tokenizer,
                 excerpt_tokens: int, test_input: str,
                 max_length: int, max_new_tokens: int) -> str:
    """
    Assemble the full k-shot CoT prompt ending with "Reasoning:".

    Layout:
        {system}

        Below are {k} examples with step-by-step reasoning.

        Example 1:
        Transcript: {short excerpt}
        Reasoning: {label-appropriate reasoning template}
        Answer: {label}

        ...

        Now analyze the following:
        Transcript: {test transcript, head+tail truncated to fit budget}
        Reasoning:

    The model will generate its reasoning, then "Answer:" is appended
    and the logit is read at that position.
    """
    header = (
        f"{system}\n\n"
        f"Below are {len(shots)} examples with step-by-step reasoning.\n"
    )

    ex_block = ""
    for i, row in enumerate(shots, 1):
        label    = row["output"].strip().lower()
        raw      = extract_transcript(row["input"])
        ids      = tokenizer(raw, add_special_tokens=False)["input_ids"][:excerpt_tokens]
        excerpt  = tokenizer.decode(ids, skip_special_tokens=True).strip()
        reasoning = _SHOT_REASONING[task][label]
        ex_block += (
            f"\nExample {i}:\n"
            f"Transcript: {excerpt}\n"
            f"Reasoning: {reasoning}\n"
            f"Answer: {label}\n"
        )

    test_header = "\nNow analyze the following:\nTranscript: "
    suffix      = "\nReasoning:"

    # Reserve tokens for: framework text + CoT generation + "\nAnswer:" (≈4 tokens)
    framework     = header + ex_block + test_header + suffix
    framework_len = len(tokenizer(framework, add_special_tokens=False)["input_ids"])
    test_budget   = max(max_length - framework_len - max_new_tokens - 4, 64)

    raw_test  = extract_transcript(test_input)
    test_ids  = head_tail_truncate(tokenizer, raw_test, test_budget)
    test_text = tokenizer.decode(test_ids, skip_special_tokens=True).strip()

    return header + ex_block + test_header + test_text + suffix


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tokenizer, prompts: list[str], labels: list[int],
            task: str, max_new_tokens: int):
    """
    For each prompt (ending with "Reasoning:"):
      1. Generate up to max_new_tokens of reasoning text.
      2. Append "\nAnswer:" to the full generated sequence.
      3. Read the logit at that final position and compare the two label tokens.
    """
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    answer_suffix_ids = tokenizer("\nAnswer:", add_special_tokens=False)["input_ids"]

    probs_pos, y_true = [], []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        # ── Step 1: generate the reasoning ───────────────────────────────────
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = {
            "input_ids":      torch.tensor([ids],            device=model.device),
            "attention_mask": torch.tensor([[1] * len(ids)], device=model.device),
        }
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                    # greedy — deterministic
            pad_token_id=tokenizer.pad_token_id,
        )

        # ── Step 2: append "\nAnswer:" and read logit ─────────────────────
        full_ids = gen_ids[0].tolist() + answer_suffix_ids
        full_enc = {
            "input_ids":      torch.tensor([full_ids],             device=model.device),
            "attention_mask": torch.tensor([[1] * len(full_ids)],  device=model.device),
        }
        logits = model(**full_enc).logits[0, -1]
        pair   = torch.tensor([logits[pos_id], logits[neg_id]])
        probs_pos.append(torch.softmax(pair, dim=0)[0].item())
        y_true.append(label)

        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(prompts))
            # Log a sample reasoning so we can inspect CoT quality in the job output
            generated_text = tokenizer.decode(
                gen_ids[0][len(ids):], skip_special_tokens=True
            )
            logger.info("  Sample reasoning: %s", generated_text[:200])

    return np.array(y_true), np.array(probs_pos)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="3-shot CoT prompting evaluation")
    ap.add_argument("--pairs",           type=Path, required=True,
                    help="JSONL pairs file (e.g. data/pairs_direction.jsonl)")
    ap.add_argument("--adapter",         type=str,  default=None,
                    help="PEFT adapter dir. Omit for base model.")
    ap.add_argument("--base-model",      default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",             type=Path, required=True,
                    help="Output JSON file for results")
    ap.add_argument("--n-shots",         type=int,  default=3)
    ap.add_argument("--excerpt-tokens",  type=int,  default=200,
                    help="Token budget per demonstration excerpt (default 200)")
    ap.add_argument("--max-length",      type=int,  default=2048,
                    help="Max tokens for the input prompt (default 2048)")
    ap.add_argument("--max-new-tokens",  type=int,  default=150,
                    help="Max tokens to generate for CoT reasoning (default 150)")
    ap.add_argument("--balance-test",    action="store_true",
                    help="Subsample majority class in test to match minority")
    ap.add_argument("--seed",            type=int,  default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = load_pairs(args.pairs)
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

    logger.info("Building %d prompts …", len(test))
    prompts, labels = [], []
    for row in test:
        prompt = build_prompt(
            system, shots, task, tokenizer,
            args.excerpt_tokens, row["input"],
            args.max_length, args.max_new_tokens,
        )
        prompts.append(prompt)
        labels.append(int(row["output"].strip().lower() == pos_word))

    logger.info("─── Sample prompt (first 800 chars) ───\n%s\n───", prompts[0][:800])

    model = load_model(args.base_model, args.adapter)
    logger.info("Running CoT inference on %d test examples …", len(prompts))
    y_true, y_prob = predict(
        model, tokenizer, prompts, labels, task, args.max_new_tokens,
    )
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
        "cot":               True,
        "max_new_tokens":    args.max_new_tokens,
        "excerpt_tokens":    args.excerpt_tokens,
        "balanced_test":     bool(args.balance_test),
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
