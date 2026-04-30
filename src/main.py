import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.loader import load_dataset, chronological_split
from features.extractor import extract_baseline_features, extract_improved_features
from models.classifier import build_classifier, train, predict
from utils.metrics import evaluate, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Earnings NLP pipeline")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--price-window", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        from data.loader import _make_synthetic_data
        import numpy as np
        df = _make_synthetic_data(args.max_samples or 200)
        df = df.rename(columns={"label": "label_direction"})
        df["price_change_pct"] = np.where(df["label_direction"] == 1, 0.02, -0.02)
        logger.info("Synthetic dataset: %d records", len(df))
    else:
        df = load_dataset(
            max_samples=args.max_samples,
            price_window=args.price_window,
            use_synthetic_fallback=True,
        )

    if df.empty:
        logger.error("Dataset is empty, aborting.")
        sys.exit(1)

    logger.info("Dataset: %d records | direction labels: %s",
                len(df), df["label_direction"].value_counts().to_dict())

    train_df, test_df = chronological_split(df, args.train_ratio)
    logger.info("Train: %d | Test: %d", len(train_df), len(test_df))

    if len(test_df) < 5:
        logger.warning("Test set is very small (%d samples), metrics may be unreliable.",
                       len(test_df))

    y_train = train_df["label_direction"].values
    y_test = test_df["label_direction"].values

    X_train_base = extract_baseline_features(train_df)
    X_test_base = extract_baseline_features(test_df)

    X_train_imp = extract_improved_features(train_df)
    X_test_imp = extract_improved_features(test_df)

    baseline_clf = train(build_classifier(), X_train_base, y_train)
    improved_clf = train(build_classifier(), X_train_imp, y_train)

    base_preds, base_probs = predict(baseline_clf, X_test_base)
    imp_preds, imp_probs = predict(improved_clf, X_test_imp)

    results = {
        "Baseline (full-transcript)": evaluate(y_test, base_preds, base_probs),
        "Improved (prepared + Q&A)": evaluate(y_test, imp_preds, imp_probs),
    }

    print_results(results)

    base_auc = results["Baseline (full-transcript)"]["auc"]
    imp_auc = results["Improved (prepared + Q&A)"]["auc"]
    if not (base_auc != base_auc):
        delta = imp_auc - base_auc
        logger.info("AUC delta (improved - baseline): %+.4f", delta)


if __name__ == "__main__":
    main()
