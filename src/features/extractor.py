"""
Feature extraction orchestration.

Baseline:  single sentiment score from full transcript
Improved:  separate scores for prepared remarks + Q&A section
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from data.preprocessor import split_sections
from features.sentiment import batch_sentiment

logger = logging.getLogger(__name__)


def extract_baseline_features(df: pd.DataFrame) -> np.ndarray:
    """
    Returns (N, 1) array: [sentiment_full]
    """
    logger.info("Extracting baseline features (full-transcript sentiment) …")
    texts  = df["text"].fillna("").tolist()
    scores = batch_sentiment(texts, desc="Baseline sentiment")
    return np.array(scores).reshape(-1, 1)


def extract_improved_features(df: pd.DataFrame) -> np.ndarray:
    """
    Returns (N, 3) array: [sentiment_full, sentiment_prepared, sentiment_qa]

    Improved model hypothesis: separating prepared-remarks sentiment
    from Q&A sentiment provides complementary signals, because:
      - Prepared remarks are management-scripted (forward-looking)
      - Q&A captures analyst skepticism / follow-up tone
    """
    logger.info("Splitting transcripts into sections …")
    sections = df["text"].fillna("").apply(split_sections)
    full_texts     = sections.apply(lambda s: s["full"]).tolist()
    prepared_texts = sections.apply(lambda s: s["prepared"]).tolist()
    qa_texts       = sections.apply(lambda s: s["qa"]).tolist()

    logger.info("Extracting improved features (full + prepared + Q&A) …")
    full_scores     = batch_sentiment(full_texts,     "Full sentiment")
    prepared_scores = batch_sentiment(prepared_texts, "Prepared sentiment")
    qa_scores       = batch_sentiment(qa_texts,       "Q&A sentiment")

    return np.column_stack([full_scores, prepared_scores, qa_scores])
