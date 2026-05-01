# Evaluation harness for Qwen2.5-7B-Instruct on earnings classification tasks
# Supports four inference modes: base-only, fine-tuned, +RAG, +few-shot (and combinations)
# When neither --rag nor --fewshot is set, the code path is identical to the original evaluate.py

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

TASK_LABELS = {
    "direction": ("up", "down"),
    "surprise": ("beat", "miss"),
}

# Delimiter used by build_pairs.py:make_prompt — must match embed_utils.DELIMITER
_DELIMITER = "\n\nTranscript:\n"

# Hard lower bound on tokens available for the main transcript after all context
_MIN_TRANSCRIPT_TOKENS = 512

def detect_task(rows: list) -> str:
    outs = {r["output"].strip().lower() for r in rows}
    if outs <= {"up", "down"}:
        return "direction"
    if outs <= {"beat", "miss"}:
        return "surprise"
    raise ValueError(f"Cannot detect task from outputs: {outs}")

def chrono_test_split(rows, train_ratio=0.9):
    n = len(rows)
    k = int(n * train_ratio)
    return rows[k:]

def first_subword_id(tokenizer, word: str) -> int:
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    assert ids, f"empty tokenization for {word!r}"
    return ids[0]

def load_model(base_model: str, adapter_dir, bf16: bool = False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
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
    rng = np.random.default_rng(seed)
    n = len(y_true)
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

# Split row["input"] into (system_instruction, transcript_text) on _DELIMITER
# Falls back to ("", full_input) if the delimiter is absent
def _split_input(input_str: str) -> Tuple[str, str]:
    parts = input_str.split(_DELIMITER, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1].strip()
    logger.warning("Delimiter %r not found in input; using full string as transcript", _DELIMITER)
    return "", input_str.strip()

# Count Qwen tokens in text (no special tokens)
def _qwen_token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

# Truncate text to the first max_tokens Qwen tokens
def _qwen_head_truncate(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

# Head+tail truncate to max_tokens Qwen tokens (mirrors finetune.py logic)
def _qwen_head_tail_truncate(tokenizer, text: str, max_tokens: int, head_ratio: float = 0.6) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head_n = int(max_tokens * head_ratio)
    tail_n = max_tokens - head_n
    ids = ids[:head_n] + ids[-tail_n:]
    return tokenizer.decode(ids, skip_special_tokens=True)

# Measure Qwen tokens consumed by fixed structural headers (independent of content)
# Called once at startup so the cost isn't re-computed for every prompt
def _measure_structural_overhead(tokenizer, has_rag: bool, has_fewshot: bool) -> int:
    headers = ""
    if has_rag:
        headers += "\n\nPrior earnings call context:\n"
    if has_fewshot:
        headers += "\n\nExample calls with known outcomes:\n"
    headers += "\n\nTranscript:\n"
    headers += "\n\nAnswer:"
    return _qwen_token_count(tokenizer, headers)

# Reduce rag_k (one at a time) then fewshot_k (two at a time) until main transcript budget >= _MIN_TRANSCRIPT_TOKENS
# Returns (main_budget, final_rag_k, final_fewshot_k)
def _budget_loop(
    max_length: int,
    overhead_tokens: int,
    system_tokens: int,
    rag_passages: list,
    rag_token_budget: int,
    fewshot_shots: list,
    fewshot_tokens_per_shot: int,
    tokenizer,
    rag_k: int,
    fewshot_k: int,
    ticker: str = "?",
    date: str = "?",
) -> Tuple[int, int, int]:
    while True:
        # Measure actual RAG token cost for the current rag_k
        if rag_k > 0 and rag_passages:
            n = min(rag_k, len(rag_passages))
            budget_per = rag_token_budget // n
            remainder = rag_token_budget - budget_per * n
            rag_parts = []
            for i, p in enumerate(rag_passages[:n]):
                p_budget = budget_per + (remainder if i == 0 else 0)
                ids = tokenizer(p["text_excerpt"], add_special_tokens=False)["input_ids"]
                if len(ids) > p_budget:
                    ids = ids[:p_budget]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                rag_parts.append(f"[{p['ticker']} | {p['date']}]\n{text}")
            rag_str = "\n\n".join(rag_parts)
            rag_tokens = _qwen_token_count(tokenizer, rag_str)
        else:
            rag_tokens = 0

        # Conservative upper bound for few-shot cost (avoids formatting all shots each iteration)
        actual_fewshot_k = min(fewshot_k, len(fewshot_shots)) if fewshot_shots else 0
        fewshot_tokens = actual_fewshot_k * fewshot_tokens_per_shot

        total_used = overhead_tokens + system_tokens + rag_tokens + fewshot_tokens
        main_budget = max_length - total_used

        if main_budget >= _MIN_TRANSCRIPT_TOKENS:
            return main_budget, rag_k, fewshot_k

        # Reduce RAG first, then few-shot (in pairs to preserve class balance)
        if rag_k > 0:
            logger.warning(
                "Budget tight for (%s, %s): main_budget=%d < %d; rag_k %d→%d",
                ticker, date, main_budget, _MIN_TRANSCRIPT_TOKENS, rag_k, rag_k - 1,
            )
            rag_k -= 1
            continue

        if fewshot_k >= 2:
            logger.warning(
                "Budget tight for (%s, %s): main_budget=%d < %d; fewshot_k %d→%d",
                ticker, date, main_budget, _MIN_TRANSCRIPT_TOKENS, fewshot_k, fewshot_k - 2,
            )
            fewshot_k -= 2
            continue

        if fewshot_k == 1:
            logger.warning(
                "Budget tight for (%s, %s): removing last fewshot shot", ticker, date,
            )
            fewshot_k = 0
            continue

        # Last resort: transcript gets everything minus fixed overhead
        main_budget = max(max_length - overhead_tokens - system_tokens, _MIN_TRANSCRIPT_TOKENS)
        logger.warning(
            "Budget exhausted all context for (%s, %s); transcript gets %d tokens",
            ticker, date, main_budget,
        )
        return main_budget, 0, 0


# Assemble the full augmented prompt: budget loop → RAG block → few-shot block → main transcript
# Returns (prompt_string, final_rag_k, final_fewshot_k)
def _assemble_prompt(
    row: dict,
    rag_passages: list,
    fewshot_shots: list,
    tokenizer,
    max_length: int,
    args,
    overhead_tokens: int,
) -> Tuple[str, int, int]:
    from retrieval.fewshot import format_shots

    system_text, main_transcript_text = _split_input(row["input"])
    system_tokens = _qwen_token_count(tokenizer, system_text)

    rag_k = min(args.rag_k, len(rag_passages)) if rag_passages else 0
    fewshot_k = min(args.fewshot_k, len(fewshot_shots)) if fewshot_shots else 0
    fewshot_tps = getattr(args, "fewshot_token_per_shot", 150)

    main_budget, final_rag_k, final_fewshot_k = _budget_loop(
        max_length=max_length,
        overhead_tokens=overhead_tokens,
        system_tokens=system_tokens,
        rag_passages=rag_passages,
        rag_token_budget=args.rag_token_budget,
        fewshot_shots=fewshot_shots,
        fewshot_tokens_per_shot=fewshot_tps,
        tokenizer=tokenizer,
        rag_k=rag_k,
        fewshot_k=fewshot_k,
        ticker=row.get("ticker", "?"),
        date=row.get("date", "?"),
    )

    # Safety clamp: never go below minimum even after the loop
    main_budget = max(main_budget, _MIN_TRANSCRIPT_TOKENS)

    prompt_parts = [system_text]

    if final_rag_k > 0 and rag_passages:
        n = min(final_rag_k, len(rag_passages))
        budget_per = args.rag_token_budget // n
        remainder = args.rag_token_budget - budget_per * n
        rag_parts = []
        for i, p in enumerate(rag_passages[:n]):
            p_budget = budget_per + (remainder if i == 0 else 0)
            text = _qwen_head_truncate(tokenizer, p["text_excerpt"], p_budget)
            rag_parts.append(f"[{p['ticker']} | {p['date']}]\n{text}")
        rag_block = "\n\n".join(rag_parts)
        prompt_parts.append(f"\n\nPrior earnings call context:\n{rag_block}")

    if final_fewshot_k > 0 and fewshot_shots:
        shots_to_use = fewshot_shots[:final_fewshot_k]
        fewshot_block = format_shots(shots_to_use, tokenizer, fewshot_tps)
        if fewshot_block:
            prompt_parts.append(f"\n\nExample calls with known outcomes:\n{fewshot_block}")

    # Main transcript truncated with head+tail to preserve opening remarks and Q&A
    truncated_main = _qwen_head_tail_truncate(tokenizer, main_transcript_text, main_budget)
    prompt_parts.append(f"\n\nTranscript:\n{truncated_main}")

    prompt = "".join(prompt_parts) + "\n\nAnswer:"

    # Hard-truncate the entire assembled prompt if it somehow still exceeds max_length
    final_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(final_ids) > max_length:
        logger.error(
            "Assembled prompt (%d tokens) exceeds max_length=%d for (%s, %s); "
            "hard-truncating entire prompt with head+tail.",
            len(final_ids), max_length, row.get("ticker"), row.get("date"),
        )
        head_n = int(max_length * 0.6)
        tail_n = max_length - head_n
        final_ids = final_ids[:head_n] + final_ids[-tail_n:]
        prompt = tokenizer.decode(final_ids, skip_special_tokens=True)

    return prompt, final_rag_k, final_fewshot_k

# Run inference over test rows; returns (y_true, y_probs_pos, rag_k_mean, fewshot_k_mean)
# When retriever and fewshot_pool are both None, this is identical to the original evaluate.py path
@torch.no_grad()
def predict(
    model,
    tokenizer,
    rows,
    task,
    max_length,
    retriever=None,
    fewshot_pool=None,
    args=None,
    overhead_tokens: int = 0,
):
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    has_rag = retriever is not None
    has_fewshot = fewshot_pool is not None

    # In-memory embedding cache keyed by (ticker, date)
    embedding_cache: dict = {}

    probs_pos: list = []
    y_true: list = []
    rag_k_log: list = []
    fewshot_k_log: list = []

    for i, row in enumerate(rows):

        if not has_rag and not has_fewshot:
            # Original code path: head_tail_truncate on the full prompt string
            prompt = row["input"] + "\n\nAnswer:"
            ids = head_tail_truncate(tokenizer, prompt, max_length)
            final_rag_k = 0
            final_fewshot_k = 0

        else:
            # Augmented code path: retrieve passages and/or few-shot examples
            rag_passages: list = []
            if has_rag:
                rag_passages = retriever.retrieve(row, embedding_cache)

            fewshot_shots: list = []
            if has_fewshot:
                from retrieval.fewshot import sample_shots
                fewshot_shots = sample_shots(
                    fewshot_pool, row, task, args.fewshot_k
                )

            prompt, final_rag_k, final_fewshot_k = _assemble_prompt(
                row=row,
                rag_passages=rag_passages,
                fewshot_shots=fewshot_shots,
                tokenizer=tokenizer,
                max_length=max_length,
                args=args,
                overhead_tokens=overhead_tokens,
            )
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        rag_k_log.append(final_rag_k)
        fewshot_k_log.append(final_fewshot_k)

        enc = {
            "input_ids": torch.tensor([ids], device=model.device),
            "attention_mask": torch.tensor([[1] * len(ids)], device=model.device),
        }
        logits = model(**enc).logits[0, -1]
        pair = torch.tensor([logits[pos_id], logits[neg_id]])
        probs_pos.append(torch.softmax(pair, dim=0)[0].item())
        y_true.append(int(row["output"].strip().lower() == pos_word))

        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(rows))

    rag_k_mean = float(np.mean(rag_k_log)) if rag_k_log else 0.0
    fewshot_k_mean = float(np.mean(fewshot_k_log)) if fewshot_k_log else 0.0

    return np.array(y_true), np.array(probs_pos), rag_k_mean, fewshot_k_mean

def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM on earnings classification tasks")

    # Original arguments (unchanged from evaluate.py).
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--balance-test", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--seed", type=int, default=42,
                    help="Master random seed (controls few-shot fallback randomness)")

    ap.add_argument("--rag", action="store_true",
                    help="Enable dense RAG retrieval")
    ap.add_argument("--index-dir", type=str, default=None,
                    help="Directory containing index.faiss and metadata.jsonl")
    ap.add_argument("--rag-k", type=int, default=3,
                    help="Target number of retrieved passages (dynamically reduced if budget requires)")
    ap.add_argument("--rag-token-budget", type=int, default=400,
                    help="Total Qwen-token budget for all retrieved passages")

    ap.add_argument("--fewshot", action="store_true",
                    help="Enable 4-shot prompting")
    ap.add_argument("--fewshot-k", type=int, default=4,
                    help="Target number of few-shot examples (dynamically reduced in pairs if budget requires)")
    ap.add_argument("--fewshot-token-per-shot", type=int, default=150,
                    help="Qwen tokens per shot transcript (head truncation)")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.rag and not args.index_dir:
        ap.error("--rag requires --index-dir")

    from transformers import AutoTokenizer

    rows = load_pairs(args.pairs)
    task = detect_task(rows)
    test = chrono_test_split(rows, 0.9)
    logger.info("Task=%s  total=%d  test=%d (before balancing)", task, len(rows), len(test))

    if args.balance_test:
        test = balance_classes(test, seed=args.seed)
        from collections import Counter
        logger.info("Test balanced: %s  n=%d",
                    dict(Counter(r["output"] for r in test)), len(test))

    tok_src = args.adapter or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    retriever = None
    if args.rag:
        from retrieval.embed_utils import load_embed_model
        from retrieval.retriever import Retriever

        # Resolve embedding model path from embed_model_path.txt or infer from index_dir.
        embed_model_path_file = Path(args.index_dir) / "embed_model_path.txt"
        if embed_model_path_file.exists():
            embed_model_path = embed_model_path_file.read_text().strip()
        else:
            embed_model_path = str(
                Path(args.index_dir).parent.parent / "models" / "bge-small-en-v1.5"
            )
            logger.warning(
                "embed_model_path.txt not found in %s; using inferred path %s",
                args.index_dir, embed_model_path,
            )

        embed_model, bge_tokenizer = load_embed_model(embed_model_path)
        retriever = Retriever(
            index_dir=args.index_dir,
            embed_model=embed_model,
            bge_tokenizer=bge_tokenizer,
            rag_k=args.rag_k,
            rag_token_budget=args.rag_token_budget,
        )
        retriever.load()

    # Build few-shot pool from the training split only (same 90/10 boundary as chrono_test_split)
    fewshot_pool = None
    if args.fewshot:
        n_all = len(rows)
        train_k = int(n_all * 0.9)
        fewshot_pool = rows[:train_k]
        logger.info(
            "Few-shot pool: %d training rows (task=%s, fewshot_k=%d, tps=%d)",
            len(fewshot_pool), task, args.fewshot_k, args.fewshot_token_per_shot,
        )

    # Measure structural overhead once; only needed when augmentation is active
    overhead_tokens = 0
    if args.rag or args.fewshot:
        overhead_tokens = _measure_structural_overhead(
            tokenizer, has_rag=args.rag, has_fewshot=args.fewshot
        )
        logger.info(
            "Structural overhead: %d Qwen tokens (rag=%s, fewshot=%s)",
            overhead_tokens, args.rag, args.fewshot,
        )

    model = load_model(args.base_model, args.adapter, bf16=args.bf16)

    y_true, y_prob, rag_k_mean, fewshot_k_mean = predict(
        model=model,
        tokenizer=tokenizer,
        rows=test,
        task=task,
        max_length=args.max_length,
        retriever=retriever,
        fewshot_pool=fewshot_pool,
        args=args,
        overhead_tokens=overhead_tokens,
    )

    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)

    pos_word, neg_word = TASK_LABELS[task]

    results = {
        "task": task,
        "pairs_file": str(args.pairs),
        "adapter": args.adapter,
        "seed": args.seed,
        "balanced_test": bool(args.balance_test),
        "n_test": int(len(y_true)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "auc": float(auc),
        "auc_ci_95": [auc_lo, auc_hi],
        "class_balance": {
            pos_word: int(y_true.sum()),
            neg_word: int(len(y_true) - y_true.sum()),
        },
        "rag": {
            "enabled": bool(args.rag),
            "index_dir": args.index_dir,
            "k_target": args.rag_k if args.rag else 0,
            "k_actual_mean": round(rag_k_mean, 3),
        },
        "fewshot": {
            "enabled": bool(args.fewshot),
            "k_target": args.fewshot_k if args.fewshot else 0,
            "tokens_per_shot": args.fewshot_token_per_shot if args.fewshot else 0,
            "k_actual_mean": round(fewshot_k_mean, 3),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
