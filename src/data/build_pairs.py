import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader import load_dataset
from data.preprocessor import strip_boilerplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIRECTION_SYSTEM = (
    "You are a financial analyst. Read the earnings call transcript below "
    "and predict whether the stock price will go UP or DOWN over the three "
    "trading days following this call. Respond with exactly one word: "
    "'up' or 'down'."
)

EPS_SYSTEM = (
    "You are a financial analyst. Read the earnings call transcript below "
    "and predict whether the company BEAT or MISSED analyst EPS consensus "
    "for this quarter. Respond with exactly one word: 'beat' or 'miss'."
)


def make_prompt(system: str, transcript: str) -> str:
    return f"{system}\n\nTranscript:\n{transcript.strip()}"


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def build_pairs(df: pd.DataFrame, out_dir: str, task: str = "both") -> None:
    direction_pairs, eps_pairs = [], []
    raw_chars, clean_chars = 0, 0

    for row in df.itertuples(index=False):
        transcript = strip_boilerplate(row.text)
        if not transcript:
            continue
        raw_chars += len(row.text)
        clean_chars += len(transcript)

        if not pd.isna(row.label_direction):
            direction_pairs.append({
                "ticker": row.ticker,
                "date": str(row.date)[:10],
                "input": make_prompt(DIRECTION_SYSTEM, transcript),
                "output": "up" if int(row.label_direction) == 1 else "down",
            })

        if not pd.isna(row.label_eps_surprise):
            eps_pairs.append({
                "ticker": row.ticker,
                "date": str(row.date)[:10],
                "reported_eps": row.reported_eps,
                "consensus_eps": row.consensus_eps,
                "input": make_prompt(EPS_SYSTEM, transcript),
                "output": "beat" if int(row.label_eps_surprise) == 1 else "miss",
            })

    if raw_chars:
        logger.info(
            "Boilerplate filter: kept %.1f%% of original chars (%d -> %d, mean %.0f -> %.0f per call)",
            100 * clean_chars / raw_chars, raw_chars, clean_chars,
            raw_chars / max(len(df), 1), clean_chars / max(len(df), 1),
        )

    if task in ("direction", "both"):
        write_jsonl(direction_pairs, os.path.join(out_dir, "pairs_direction.jsonl"))
    if task in ("surprise", "both"):
        write_jsonl(eps_pairs, os.path.join(out_dir, "pairs_eps_surprise.jsonl"))

    if direction_pairs:
        up_pct = sum(p["output"] == "up" for p in direction_pairs) / len(direction_pairs) * 100
        logger.info("Direction pairs: %d total | up=%.1f%% down=%.1f%%",
                    len(direction_pairs), up_pct, 100 - up_pct)
    if eps_pairs:
        beat_pct = sum(p["output"] == "beat" for p in eps_pairs) / len(eps_pairs) * 100
        logger.info("EPS pairs: %d total | beat=%.1f%% miss=%.1f%%",
                    len(eps_pairs), beat_pct, 100 - beat_pct)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--price-window", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="data")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--task", choices=["direction", "surprise", "both"], default="both")
    p.add_argument("--min-date", type=str, default=None)
    p.add_argument("--newest-first", action="store_true")
    p.add_argument("--delay-sec", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Loading dataset (max_samples=%d task=%s min_date=%s newest_first=%s delay=%.2fs)",
        args.max_samples, args.task, args.min_date, args.newest_first, args.delay_sec,
    )

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
            use_synthetic_fallback=False,
            task=args.task,
            min_date=args.min_date,
            newest_first=args.newest_first,
            delay_sec=args.delay_sec,
        )

    logger.info("Loaded %d records", len(df))
    build_pairs(df, args.out_dir, task=args.task)


if __name__ == "__main__":
    main()
