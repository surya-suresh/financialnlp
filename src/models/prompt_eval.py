"""
prompt_eval.py — 4-shot few-shot prompting evaluation (base or finetuned model).

Each few-shot demonstration shows a financially relevant transcript excerpt and
its known outcome. Shot excerpts are built via extractive sentence scoring rather
than head+tail truncation, avoiding boilerplate (operator intros, legal disclaimers).
Shots are strictly 2+2 class-balanced and ordered to alternate labels, neutralising
recency bias. The model predicts via logit comparison at the "Answer:" position —
identical to evaluate.py so results are directly comparable.

Run:
    # base model, 4-shot, direction
    python -m src.models.prompt_eval \\
        --pairs data/pairs_direction.jsonl \\
        --out   outputs/direction/results_prompt_base.json

    # finetuned model, 4-shot, direction
    python -m src.models.prompt_eval \\
        --pairs   data/pairs_direction.jsonl \\
        --adapter outputs/direction/adapter \\
        --out     outputs/direction/results_prompt_ft.json

    # base model, 4-shot, surprise
    python -m src.models.prompt_eval \\
        --pairs data/pairs_eps_surprise.jsonl \\
        --out   outputs/surprise/results_prompt_base.json

    # finetuned model, 4-shot, surprise
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
    Select n_shots with strict 2+2 class balance and alternating label order.

    Each class gets n_shots//2 examples drawn from evenly-spaced temporal
    segments of that class's training rows, ensuring temporal diversity within
    each class. Shots are interleaved (pos, neg, pos, neg, ...) so the label
    immediately before the test example is never systematically biased toward
    one class, mitigating recency bias in the logit.

    Requires n_shots to be even.
    """
    if n_shots % 2 != 0:
        raise ValueError(f"n_shots must be even for strict 2+2 balance; got {n_shots}")

    rng = random.Random(seed)
    half = n_shots // 2

    def pick_from_class(rows: list[dict], k: int) -> list[dict]:
        """Pick k examples spread across k equal temporal segments."""
        n = len(rows)
        if n == 0:
            return []
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

    # Interleave: pos, neg, pos, neg, ...
    shots = []
    for p, n in zip(pos_shots, neg_shots):
        shots.append(p)
        shots.append(n)
    return shots


# Keywords used to score sentence financial relevance for shot excerpts.
# Deliberately excludes outcome words ("beat", "missed", "exceeded") to prevent
# the scorer from selecting sentences that restate the label rather than signal it.
# Sorted longest-first so _score_sentence can blank matched spans and avoid
# double-counting sub-keywords (e.g. "per share" inside "earnings per share").
_FINANCIAL_KEYWORDS = sorted([
    "earnings per share", "operating income", "net income", "per share",
    "eps", "revenue", "guidance", "outlook", "raised", "lowered",
    "below", "above", "margin", "growth", "decline", "consensus",
    "diluted", "forecast", "expect", "quarter", "fiscal", "sales",
    "profit", "loss",
], key=len, reverse=True)


def _score_sentence(sentence: str) -> int:
    """
    Count distinct financial keywords in a sentence.  Longer keywords are
    matched first and their span is blanked out so shorter sub-keywords
    (e.g. "per share" inside "earnings per share") are not double-counted.
    Keywords are sorted longest-first at module load via _FINANCIAL_KEYWORDS.
    """
    remaining = sentence.lower()
    score = 0
    for kw in _FINANCIAL_KEYWORDS:  # already sorted longest-first
        if kw in remaining:
            score += 1
            remaining = remaining.replace(kw, " " * len(kw))
    return score


def extract_top_sentences(transcript: str, tokenizer, budget: int) -> str:
    """
    Score each sentence by financial keyword density, then greedily pack the
    highest-scoring sentences into the token budget.  Returns them joined in
    their original document order so the excerpt reads coherently.

    Sentence splitting uses a regex that avoids breaking on decimal numbers
    (e.g. "$2.18") and common abbreviations, splitting only when a period is
    followed by whitespace and is not surrounded by digits.
    """
    cleaned = transcript.replace("\n", " ")
    # Split on ". " only when the period is not adjacent to a digit on either side
    raw_sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)\s+', cleaned)
                     if len(s.strip()) > 20]

    scored = []
    for s in raw_sentences:
        score = _score_sentence(s)
        scored.append((score, s))

    # Sort descending by score; keep original index for order restoration
    ranked = sorted(enumerate(scored), key=lambda x: x[1][0], reverse=True)

    chosen_indices, total_tokens = [], 0
    for orig_idx, (score, sentence) in ranked:
        if score == 0:
            break  # no financial keywords — stop adding sentences
        n_tok = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if total_tokens + n_tok > budget:
            continue
        chosen_indices.append(orig_idx)
        total_tokens += n_tok

    if not chosen_indices:
        # Fallback: take the first sentence that fits if nothing was scored
        for sentence in raw_sentences:
            n_tok = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
            if n_tok <= budget:
                return sentence
        return ""

    # Restore document order
    chosen_indices.sort()
    return ". ".join(scored[i][1] for i in chosen_indices)


def build_shot_excerpts(shots: list[dict], tokenizer, excerpt_tokens: int) -> list[str]:
    """
    Pre-build extractive excerpts for all shots once.  Called once before the
    test loop so the same shot transcripts are not re-tokenized per test example.
    """
    excerpts = []
    for row in shots:
        raw     = extract_transcript(row["input"])
        excerpt = extract_top_sentences(raw, tokenizer, excerpt_tokens)
        excerpts.append(excerpt)
    return excerpts


def build_prompt(system: str, shots: list[dict], shot_excerpts: list[str],
                 shot_context_blocks: list[str],
                 tokenizer, test_input: str, max_length: int,
                 context_block: str = "", rag_prefix: str = "") -> str:
    """
    Assemble the full k-shot prompt ending with "Answer:".

    Accepts pre-built shot excerpts (from build_shot_excerpts) and per-shot
    context blocks so that demonstrations and the test example share the same
    format — both show structured context before the transcript when available.

    The test transcript is built using the same extractive sentence scoring as
    the shot excerpts, ensuring consistent text representation across the full
    prompt. Falls back to head+tail truncation if extraction yields nothing.

    After assembly the prompt is verified to fit within max_length by
    iteratively shrinking the test transcript budget if retokenization drift
    causes an overrun — ensuring "Answer:" is always the final token sequence
    and the logit is read at the correct position.

    Layout (with RAG and context blocks):
        {rag_prefix}            ← retrieved prior transcripts (omitted when empty)

        {system}

        Below are {k} examples with known outcomes.

        Example 1:
        {shot context block}   ← same structured features as test
        Transcript: {extractive excerpt}
        Answer: {label}

        ...

        Now analyze the following:
        {test context block}
        Transcript: {extractive excerpt, budget = remaining tokens}
        Answer:
    """
    rag_section = f"{rag_prefix}\n\n" if rag_prefix else ""
    header = (
        f"{rag_section}"
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

    # Context block (non-transcript numerical features) injected before transcript
    ctx_section = f"\n{context_block}\n" if context_block else ""
    test_header = f"\nNow analyze the following:{ctx_section}\nTranscript: "
    suffix      = "\nAnswer:"

    # Compute remaining token budget for the test transcript
    framework     = header + ex_block + test_header + suffix
    framework_len = len(tokenizer(framework, add_special_tokens=False)["input_ids"])
    test_budget   = max(max_length - framework_len, 64)

    raw_test = extract_transcript(test_input)

    # Use the same extractive scoring as shots so the text representation is
    # consistent across demonstrations and the test example.  Fall back to
    # head+tail only if extraction yields nothing (e.g. non-English transcript).
    # Iteratively tighten budget to guard against detokenise→retokenise drift.
    for _ in range(20):
        test_text = extract_top_sentences(raw_test, tokenizer, test_budget)
        if not test_text:
            ids = head_tail_truncate(tokenizer, raw_test, test_budget)
            test_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        full_prompt = header + ex_block + test_header + test_text + suffix
        if len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"]) <= max_length:
            break
        test_budget = max(test_budget - 16, 64)

    return full_prompt


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tokenizer, prompts: list[str], labels: list[int],
            task: str, max_length: int):
    """
    Read the logit at the final "Answer:" position and compare the two
    target tokens — same method as evaluate.py for direct comparability.

    Verifies that each prompt fits within max_length so the "Answer:" suffix
    is never truncated — which would cause the logit to be read at the wrong
    position and produce silently meaningless predictions.
    """
    pos_word, neg_word = TASK_LABELS[task]
    pos_id = first_subword_id(tokenizer, pos_word)
    neg_id = first_subword_id(tokenizer, neg_word)

    probs_pos, y_true = [], []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            raise RuntimeError(
                f"Prompt {i} has {len(ids)} tokens which exceeds max_length={max_length}. "
                f"The 'Answer:' suffix would be truncated, invalidating the logit read. "
                f"Reduce --excerpt-tokens or --max-length."
            )
        tail_text = tokenizer.decode(ids[-10:], skip_special_tokens=True)
        if not tail_text.rstrip().endswith("Answer:"):
            raise RuntimeError(
                f"Prompt {i} does not end with 'Answer:'. "
                f"Tail decode: {tail_text!r}"
            )
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

def _load_tokenizer(tok_src: str):
    """
    Load tokenizer with exponential-backoff retry for transient NFS ESTALE
    errors (errno 116 — "Stale file handle") common on OSC/HPC shared
    filesystems when reading tokenizer JSON files.

    Retries up to 3 times with delays of 10 s, 20 s, 30 s before re-raising.
    """
    import time
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(3):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as exc:
            is_stale = (
                "Stale file handle" in str(exc)
                or getattr(exc, "errno", None) == 116
                or "os error 116" in str(exc).lower()
            )
            if is_stale and attempt < 2:
                delay = 10 * (attempt + 1)
                logger.warning(
                    "Tokenizer load attempt %d/3 failed (stale file handle): %s"
                    " — retrying in %d s …", attempt + 1, exc, delay,
                )
                last_exc = exc
                time.sleep(delay)
            else:
                raise
    raise last_exc


def main():
    ap = argparse.ArgumentParser(description="4-shot few-shot prompting evaluation")
    ap.add_argument("--pairs",          type=Path, required=True,
                    help="JSONL pairs file (e.g. data/pairs_direction.jsonl)")
    ap.add_argument("--adapter",        type=str,  default=None,
                    help="PEFT adapter dir. Omit for base model.")
    ap.add_argument("--base-model",     default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",            type=Path, required=True,
                    help="Output JSON file for results")
    ap.add_argument("--n-shots",        type=int,  default=4)
    ap.add_argument("--excerpt-tokens", type=int,  default=150,
                    help="Token budget per demonstration excerpt (default 150)")
    ap.add_argument("--max-length",     type=int,  default=3072)
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
    ap.add_argument("--rag",                action="store_true",
                    help="Enable retrieval-augmented generation: prepend prior transcripts "
                         "from the same company before the system prompt and few-shot block.")
    ap.add_argument("--rag-top-k",          type=int, default=2,
                    help="Number of passages to retrieve per example (default 2).")
    ap.add_argument("--rag-context-tokens", type=int, default=400,
                    help="Total token budget reserved for the RAG context block "
                         "(approx; default 400).  Each passage gets an equal share.")
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
    tokenizer = _load_tokenizer(tok_src)

    system = TASK_SYSTEMS[task]

    # Build RAG retriever from training split (once; reused for every test example).
    retriever             = None
    rag_chars_per_passage = 800
    if args.rag:
        from retrieval.retriever import TranscriptRetriever
        logger.info("Building RAG retriever from training split …")
        retriever = TranscriptRetriever.from_pairs(rows, train_ratio=0.9)
        rag_chars_per_passage = max(
            200, args.rag_context_tokens * 4 // max(args.rag_top_k, 1)
        )
        logger.info("RAG retriever ready (corpus=%d docs).", len(retriever.corpus))

    # Build shot excerpts once — avoids re-tokenising the same transcripts
    # for every test example inside the prompt-building loop.
    logger.info("Pre-building shot excerpts …")
    shot_excerpts = build_shot_excerpts(shots, tokenizer, args.excerpt_tokens)

    # Pre-fetch context blocks for both shots and test in a single batch so
    # demonstrations and the test example share the same prompt format.
    context_map: dict = {}
    shot_context_blocks: list[str] = [""] * len(shots)
    if args.context:
        logger.info("Fetching non-transcript context blocks via yfinance …")
        context_map = fetch_context_blocks_batch(shots + test)
        n_filled = sum(1 for v in context_map.values() if v)
        logger.info("Context blocks: %d / %d non-empty", n_filled, len(shots) + len(test))
        shot_context_blocks = [
            context_map.get((s["ticker"], s["date"][:10]), "") for s in shots
        ]

    logger.info("Building %d prompts …", len(test))
    prompts, labels = [], []
    for row in test:
        ctx_block = context_map.get((row["ticker"], row["date"][:10]), "")
        rag_prefix_str = ""
        if retriever is not None:
            from retrieval.retriever import build_rag_prefix
            transcript     = extract_transcript(row["input"])
            passages       = retriever.retrieve(
                row["ticker"], row["date"], transcript, k=args.rag_top_k,
            )
            rag_prefix_str = build_rag_prefix(
                passages, max_chars_per_passage=rag_chars_per_passage,
            )
        prompt = build_prompt(
            system, shots, shot_excerpts, shot_context_blocks, tokenizer,
            row["input"], args.max_length,
            context_block=ctx_block,
            rag_prefix=rag_prefix_str,
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
        "task":               task,
        "pairs_file":         str(args.pairs),
        "adapter":            args.adapter,
        "n_shots":            args.n_shots,
        "seed":               args.seed,
        "max_length":         args.max_length,
        "cot":                False,
        "context":            bool(args.context),
        "rag":                bool(args.rag),
        "rag_top_k":          args.rag_top_k if args.rag else 0,
        "rag_context_tokens": args.rag_context_tokens if args.rag else 0,
        "excerpt_tokens":     args.excerpt_tokens,
        "balanced_test":      bool(args.balance_test),
        "min_eps_margin":     float(args.min_eps_margin),
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
