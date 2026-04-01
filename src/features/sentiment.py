"""
FinBERT-based sentiment feature extraction.

Outputs a scalar sentiment score in [-1, +1]:
    score = P(positive) - P(negative)

Processes text in chunks to handle long transcripts that exceed
FinBERT's 512-token limit.
"""

import logging
import math
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_MAX_TOKENS  = 450   # leave a small margin below 512
_CHUNK_WORDS = 150   # approx words per chunk (conservative)

# Module-level cache so tokenizer/model are loaded once
_tokenizer = None
_model     = None
_device    = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Loading FinBERT from %s …", _MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
        logger.info("FinBERT loaded on %s", _device)
    except Exception as e:
        logger.error("Failed to load FinBERT: %s", e)
        raise


def _fallback_to_cpu():
    """Move model to CPU after a CUDA kernel incompatibility error."""
    global _device
    logger.warning(
        "CUDA kernel error detected (likely GPU/PyTorch CC mismatch). "
        "Falling back to CPU for inference."
    )
    _device = "cpu"
    _model.to("cpu")


def _chunk_text(text: str) -> List[str]:
    """Split text into word-level chunks that fit within the token limit."""
    words  = text.split()
    chunks = []
    for i in range(0, max(len(words), 1), _CHUNK_WORDS):
        chunk = " ".join(words[i: i + _CHUNK_WORDS])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text[:500]]


def _score_chunks(chunks: List[str]) -> float:
    """
    Score a list of text chunks with FinBERT.
    Returns the mean (P_positive - P_negative) across chunks.
    Chunks weighted equally (could weight by length, kept simple).
    """
    import torch
    import torch.nn.functional as F

    scores = []
    with torch.no_grad():
        for chunk in chunks:
            enc = _tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_TOKENS,
                padding=True,
            )
            enc = {k: v.to(_device) for k, v in enc.items()}
            try:
                logits = _model(**enc).logits  # shape (1, 3)
            except (RuntimeError, Exception) as e:
                if "no kernel image" in str(e) or "CUDA" in str(e):
                    _fallback_to_cpu()
                    enc    = {k: v.cpu() for k, v in enc.items()}
                    logits = _model(**enc).logits
                else:
                    raise
            probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            # FinBERT label order: positive=0, negative=1, neutral=2
            # (confirmed from ProsusAI/finbert model card)
            score  = float(probs[0]) - float(probs[1])
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def sentiment_score(text: str) -> float:
    """
    Return a single sentiment score for an arbitrary-length text.
    Score in [-1, +1]: positive values = bullish, negative = bearish.
    Returns 0.0 if text is empty.
    """
    if not text or not text.strip():
        return 0.0
    _load_model()
    chunks = _chunk_text(text)
    return _score_chunks(chunks)


def batch_sentiment(texts: List[str],
                    desc: str = "Scoring") -> List[float]:
    """
    Score a list of texts, logging progress every 10%.
    """
    _load_model()
    n      = len(texts)
    scores = []
    log_every = max(1, n // 10)
    for i, text in enumerate(texts):
        if i % log_every == 0:
            logger.info("  %s: %d / %d", desc, i, n)
        scores.append(sentiment_score(text))
    return scores
