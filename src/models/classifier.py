import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight="balanced")),
    ])


def train(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def predict(pipeline: Pipeline, X: np.ndarray):
    labels = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    return labels, probs
