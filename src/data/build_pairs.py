"""
build_pairs.py — Export LLM finetuning pairs for both prediction tasks.

Each output file is newline-delimited JSON (JSONL).  Every line is one
training example with two keys:
    "input"  — the prompt text fed to the model
    "output" — the single-token label the model must produce

Output files
------------
    <out_dir>/pairs_direction.jsonl      (transcript → "up" / "down")
    <out_dir>/pairs_eps_surprise.jsonl   (transcript → "beat" / "miss")

Each file contains only rows for which that particular label is available
(yahooquery may not have EPS data for every ticker/date).

Usage
-----
    python src/data/build_pairs.py [--max-samples N] [--out-dir DIR] [--synthetic]
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader import load_dataset, chronological_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DIRECTION_SYSTEM = (
    "You are a financial analyst. Read the earnings call transcript below "
    "and predict whether the stock price will go UP or DOWN over the three "
    "trading days following this call. Respond with exactly one word: "
    "'up' or 'down'."
)

_EPS_SYSTEM = (
    "You are a financial analyst. Read the earnings call transcript below "
    "and predict whether the company BEAT or MISSED analyst EPS consensus "
    "for this quarter. Respond with exactly one word: 'beat' or 'miss'."
)


def _make_prompt(system: str, transcript: str) -> str:
    return f"{system}\n\nTranscript:\n{transcript.strip()}"


# ---------------------------------------------------------------------------
# Label converters
# ---------------------------------------------------------------------------

def _direction_label(value: int) -> str:
    return "up" if value == 1 else "down"


def _eps_label(value: int) -> str:
    return "beat" if value == 1 else "miss"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pairs(df: pd.DataFrame, out_dir: str) -> None:
    direction_pairs = []
    eps_pairs       = []

    for row in df.itertuples(index=False):
        transcript = row.text

        # --- stock direction ---
        if not pd.isna(row.label_direction):
            direction_pairs.append({
                "ticker": row.ticker,
                "date":   str(row.date)[:10],
                "input":  _make_prompt(_DIRECTION_SYSTEM, transcript),
                "output": _direction_label(int(row.label_direction)),
            })

        # --- EPS surprise ---
        if not pd.isna(row.label_eps_surprise):
            eps_pairs.append({
                "ticker":        row.ticker,
                "date":          str(row.date)[:10],
                "reported_eps":  row.reported_eps,
                "consensus_eps": row.consensus_eps,
                "input":         _make_prompt(_EPS_SYSTEM, transcript),
                "output":        _eps_label(int(row.label_eps_surprise)),
            })

    write_jsonl(direction_pairs, os.path.join(out_dir, "pairs_direction.jsonl"))
    write_jsonl(eps_pairs,       os.path.join(out_dir, "pairs_eps_surprise.jsonl"))

    # Summary statistics
    if direction_pairs:
        dir_df  = pd.DataFrame(direction_pairs)
        up_pct  = (dir_df["output"] == "up").mean() * 100
        logger.info("Direction pairs: %d total | up=%.1f%%  down=%.1f%%",
                    len(dir_df), up_pct, 100 - up_pct)

    if eps_pairs:
        eps_df   = pd.DataFrame(eps_pairs)
        beat_pct = (eps_df["output"] == "beat").mean() * 100
        logger.info("EPS pairs:       %d total | beat=%.1f%%  miss=%.1f%%",
                    len(eps_df), beat_pct, 100 - beat_pct)


def parse_args():
    parser = argparse.ArgumentParser(description="Build LLM finetuning pairs")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max transcripts to load (default 500)")
    parser.add_argument("--price-window", type=int, default=3,
                        help="Trading days after call for price label (default 3)")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (smoke test, no internet needed)")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading dataset (max_samples=%d) …", args.max_samples)

    if args.synthetic:
        from data.loader import _make_synthetic_data
        import numpy as np
        df = _make_synthetic_data(args.max_samples or 200)
        df = df.rename(columns={"label": "label_direction"})
        df["price_change_pct"] = np.where(df["label_direction"] == 1, 0.02, -0.02)
    else:
        df = load_dataset(
            max_samples=args.max_samples,
            price_window=args.price_window,
            use_synthetic_fallback=True,
        )

    logger.info("Loaded %d records", len(df))

    build_pairs(df, args.out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
