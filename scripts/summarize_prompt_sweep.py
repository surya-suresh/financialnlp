#!/usr/bin/env python3
"""Summarize prompt seed-sweep JSON files."""

import argparse
import glob
import json
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prompt seed sweep outputs")
    parser.add_argument("pattern", help="Glob pattern, e.g. outputs/surprise/results_prompt_ft_v3_seed*.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.pattern)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "path": path,
            "seed": data.get("seed"),
            "auc": data.get("auc"),
            "balanced_accuracy": data.get("balanced_accuracy"),
            "accuracy": data.get("accuracy"),
            "n_test": data.get("n_test"),
        })

    if not rows:
        raise SystemExit(f"No files matched: {args.pattern}")

    aucs = [r["auc"] for r in rows if r["auc"] is not None]
    baccs = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
    best = max(rows, key=lambda r: r["auc"] if r["auc"] is not None else float("-inf"))

    print(f"files: {len(rows)}")
    print(f"n_test values: {sorted(set(r['n_test'] for r in rows))}")
    print(f"auc mean/std/min/max: {mean(aucs):.4f} / {pstdev(aucs):.4f} / {min(aucs):.4f} / {max(aucs):.4f}")
    print(f"balanced acc mean: {mean(baccs):.4f}")
    print(f"best seed: {best['seed']}  auc: {best['auc']:.4f}  file: {best['path']}")


if __name__ == "__main__":
    main()
