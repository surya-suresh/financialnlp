"""
main.py — End-to-end earnings call sentiment → stock prediction pipeline.

Run:
    python src/main.py [--max-samples N] [--synthetic]

Steps:
    1. Load earnings call transcripts + stock price labels
    2. Chronological train/test split
    3. Extract baseline features  (full-transcript FinBERT sentiment)
    4. Extract improved features  (prepared-remarks + Q&A FinBERT sentiment)
    5. Train logistic regression for each feature set
    6. Evaluate and print results table
"""

import argparse
import logging
import sys
import os

# Allow imports from src/ regardless of where the script is invoked
sys.path.insert(0, os.path.dirname(__file__))

from data.loader      import load_dataset, chronological_split
from features.extractor import extract_baseline_features, extract_improved_features
from models.classifier  import build_classifier, train, predict
from utils.metrics      import evaluate, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Earnings NLP pipeline")
    parser.add_argument("--max-samples", type=int, default=300,
                        help="Max number of earnings calls to load (default 300)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data (smoke test, no internet needed)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of data for training (default 0.8)")
    parser.add_argument("--price-window", type=int, default=3,
                        help="Trading days after call for price label (default 3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading data")
    logger.info("=" * 60)

    if args.synthetic:
        # Bypass HuggingFace entirely
        from data.loader import _make_synthetic_data
        import numpy as np
        df = _make_synthetic_data(args.max_samples or 200)
        df["price_change"] = np.where(df["label"] == 1, 0.02, -0.02)
        df = df[["ticker", "date", "text", "label", "price_change"]]
        logger.info("Synthetic dataset: %d records", len(df))
    else:
        df = load_dataset(
            max_samples=args.max_samples,
            price_window=args.price_window,
            use_synthetic_fallback=True,   # auto-fallback if HF unavailable
        )

    if df.empty:
        logger.error("Dataset is empty — aborting.")
        sys.exit(1)

    logger.info("Dataset: %d records  |  labels: %s",
                len(df), df["label"].value_counts().to_dict())

    # ------------------------------------------------------------------
    # 2. Chronological split
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 — Chronological train/test split (%.0f%% / %.0f%%)",
                args.train_ratio * 100, (1 - args.train_ratio) * 100)
    logger.info("=" * 60)

    train_df, test_df = chronological_split(df, args.train_ratio)
    logger.info("Train: %d  |  Test: %d", len(train_df), len(test_df))

    if len(test_df) < 5:
        logger.warning("Test set is very small (%d samples) — "
                       "metrics may be unreliable.", len(test_df))

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # ------------------------------------------------------------------
    # 3 & 4. Feature extraction
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 — Baseline feature extraction (full transcript)")
    logger.info("=" * 60)

    X_train_base = extract_baseline_features(train_df)
    X_test_base  = extract_baseline_features(test_df)

    logger.info("=" * 60)
    logger.info("STEP 4 — Improved feature extraction (prepared + Q&A)")
    logger.info("=" * 60)

    X_train_imp = extract_improved_features(train_df)
    X_test_imp  = extract_improved_features(test_df)

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 — Training logistic regression classifiers")
    logger.info("=" * 60)

    baseline_clf = train(build_classifier(), X_train_base, y_train)
    improved_clf = train(build_classifier(), X_train_imp,  y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6 — Evaluation")
    logger.info("=" * 60)

    base_preds, base_probs = predict(baseline_clf, X_test_base)
    imp_preds,  imp_probs  = predict(improved_clf,  X_test_imp)

    results = {
        "Baseline (full-transcript)": evaluate(y_test, base_preds, base_probs),
        "Improved (prepared + Q&A)":  evaluate(y_test, imp_preds,  imp_probs),
    }

    print_results(results)

    # Log a brief interpretation
    base_auc = results["Baseline (full-transcript)"]["auc"]
    imp_auc  = results["Improved (prepared + Q&A)"]["auc"]
    if not (base_auc != base_auc):  # not NaN
        delta = imp_auc - base_auc
        direction = "improvement" if delta >= 0 else "regression"
        logger.info("AUC delta (improved − baseline): %+.4f  [%s]",
                    delta, direction)


if __name__ == "__main__":
    main()
