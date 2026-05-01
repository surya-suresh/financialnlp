# Few-shot example sampling for RAG evaluation
# sample_shots selects up to k training examples balanced across labels, most-recent-first
# format_shots renders the selected examples into a prompt block
# All operations are deterministic (date-sort only, no randomness)

import logging
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embed_utils import extract_raw_transcript_text

logger = logging.getLogger(__name__)

TASK_LABELS = {
    "direction": ("up", "down"),
    "surprise": ("beat", "miss"),
}

# Select up to k shots from train_rows: balanced positive/negative, most-recent-first
# Falls back to the k most-recent eligible rows if one class is entirely absent
def sample_shots(
    train_rows: List[dict],
    query_row: dict,
    task: str,
    k: int,
) -> List[dict]:
    if k <= 0:
        return []

    pos_word, neg_word = TASK_LABELS[task]
    query_date = query_row["date"]

    # Eligible pool: strict less-than on ISO date strings (YYYY-MM-DD sorts lexicographically)
    eligible = [r for r in train_rows if r["date"] < query_date]

    if not eligible:
        logger.debug(
            "No eligible shots for query (ticker=%s, date=%s): "
            "query predates all training rows.",
            query_row.get("ticker"), query_date,
        )
        return []

    pos_pool = [r for r in eligible if r["output"] == pos_word]
    neg_pool = [r for r in eligible if r["output"] == neg_word]

    # Reverse to most-recent-first within each pool
    pos_pool = pos_pool[::-1]
    neg_pool = neg_pool[::-1]

    n_per_class = k // 2

    selected_pos = pos_pool[:n_per_class]
    selected_neg = neg_pool[:n_per_class]

    pos_taken = len(selected_pos)
    neg_taken = len(selected_neg)
    pos_shortfall = n_per_class - pos_taken
    neg_shortfall = n_per_class - neg_taken

    # Fill shortfall from the other class if one pool runs short
    if pos_shortfall > 0:
        extra = neg_pool[neg_taken : neg_taken + pos_shortfall]
        selected_neg = selected_neg + extra

    if neg_shortfall > 0:
        extra = pos_pool[pos_taken : pos_taken + neg_shortfall]
        selected_pos = selected_pos + extra

    # Interleave [pos_0, neg_0, pos_1, neg_1, ...] and cap at k
    shots: List[dict] = []
    for i in range(max(len(selected_pos), len(selected_neg))):
        if i < len(selected_pos):
            shots.append(selected_pos[i])
        if i < len(selected_neg) and len(shots) < k:
            shots.append(selected_neg[i])
    shots = shots[:k]

    # Last-resort fallback: use k most-recent eligible rows if both pools were empty
    if len(shots) == 0 and eligible:
        shots = eligible[-k:][::-1]
        logger.warning(
            "Shot pools were empty after class-balanced selection "
            "(ticker=%s, date=%s); using %d most-recent eligible rows as fallback.",
            query_row.get("ticker"), query_date, len(shots),
        )

    return shots


# Format a list of shot rows into the few-shot prompt block
# Each shot is head-truncated to tokens_per_shot Qwen tokens; shots are separated by blank lines
def format_shots(
    shots: List[dict],
    qwen_tokenizer,
    tokens_per_shot: int,
) -> str:
    if not shots:
        return ""

    parts = []
    for shot_row in shots:
        shot_text = extract_raw_transcript_text(shot_row)
        ids = qwen_tokenizer(shot_text, add_special_tokens=False)["input_ids"]
        if len(ids) > tokens_per_shot:
            ids = ids[:tokens_per_shot]
            shot_text = qwen_tokenizer.decode(ids, skip_special_tokens=True)
        parts.append(f"Transcript:\n{shot_text}\nAnswer: {shot_row['output']}")

    return "\n\n".join(parts)
