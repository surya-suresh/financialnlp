import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc}


def print_results(results: dict) -> None:
    header = f"{'Model':<30}{'Accuracy':>12}{'AUC-ROC':>12}"
    divider = "-" * len(header)
    print()
    print(divider)
    print(header)
    print(divider)
    for name, m in results.items():
        acc = m.get("accuracy", float("nan"))
        auc = m.get("auc", float("nan"))
        print(f"{name:<30}{acc:>12.4f}{auc:>12.4f}")
    print(divider)
    print()
