"""
src/retrieval/retriever.py

Modular retrieval module for RAG-augmented inference over earnings call transcripts.

Design
------
TranscriptRetriever builds its corpus exclusively from the training split
(same 90/10 chronological split used across the pipeline) so test labels
never leak into retrieved context.

Retrieval strategy (in priority order):
  1. Company-prior: most-recent training transcripts from the same ticker
     with date strictly before the query date.  This lets the model compare
     how management tone, guidance, and analyst concerns have changed quarter
     over quarter.
  2. BM25 fallback: remaining slots are filled with top-ranked documents
     from the rest of the corpus (different tickers), giving the model
     general domain context when company-specific history is sparse.

BM25 is implemented without external dependencies (stdlib only) so no new
packages are required.  The class API is deliberately simple so it can be
swapped for a dense retriever (sentence-transformers + FAISS) without
touching the calling code.

Public API
----------
    rows = load_pairs(pairs_path)               # sorted by date (as in finetune.py)
    retriever = TranscriptRetriever.from_pairs(rows, train_ratio=0.9)
    passages  = retriever.retrieve(ticker, date, query_text, k=2)
    prefix    = build_rag_prefix(passages, max_chars_per_passage=800)
    # prepend `prefix` to the prompt before inference
"""

import logging
import math
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_transcript(input_text: str) -> str:
    """Pull the transcript body out of the stored prompt string."""
    for marker in ("Transcript:\n", "Transcript:"):
        if marker in input_text:
            return input_text.split(marker, 1)[-1].strip()
    return input_text.strip()


def _tokenize(text: str) -> list[str]:
    """Minimal whitespace tokenizer for BM25 (no external deps)."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _quarter_label(date_str: str) -> str:
    """Return 'Q1 YYYY' style label from a YYYY-MM-DD string."""
    try:
        month = int(date_str[5:7])
        year  = date_str[:4]
        return f"Q{(month - 1) // 3 + 1} {year}"
    except (IndexError, ValueError):
        return date_str


# ---------------------------------------------------------------------------
# BM25 (self-contained, no extra packages)
# ---------------------------------------------------------------------------

class _BM25:
    """
    Okapi BM25 ranking over a pre-tokenised corpus.
    Requires only Python stdlib (math, re).
    """

    def __init__(self, corpus_tokens: list[list[str]],
                 k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.n  = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.n, 1)
        self._corpus = corpus_tokens

        df: dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self._idf: dict[str, float] = {
            t: math.log((self.n - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def top_n(self, query_tokens: list[str], n: int) -> list[int]:
        """Return indices of the top-n scoring documents."""
        scores = [0.0] * self.n
        for term in query_tokens:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for i, doc in enumerate(self._corpus):
                tf = doc.count(term)
                dl = len(doc)
                scores[i] += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return sorted(range(self.n), key=lambda i: scores[i], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Public retriever
# ---------------------------------------------------------------------------

class TranscriptRetriever:
    """
    Retrieves relevant prior earnings call passages for a given query.

    The corpus is built from the training split only (train_ratio controls
    the same 90/10 cut used throughout the pipeline).

    Retrieval strategy:
      1. Company-prior  — same ticker, most-recent dates < query date
      2. BM25 fallback  — top-ranked docs from other tickers (fills remaining slots)

    The class is designed to be swap-compatible: replace ``from_pairs`` and
    ``retrieve`` with a dense-embedding implementation (sentence-transformers
    + FAISS) without changing any calling code.
    """

    def __init__(self, corpus: list[dict]):
        """
        Parameters
        ----------
        corpus : list of dicts with keys:
            ticker  (str)
            date    (str, "YYYY-MM-DD")
            text    (str, transcript excerpt for retrieval)
        """
        self.corpus = corpus

        # Group by ticker so company-prior lookups are O(|company docs|).
        self._by_ticker: dict[str, list[dict]] = {}
        for doc in corpus:
            self._by_ticker.setdefault(doc["ticker"], []).append(doc)
        for t in self._by_ticker:
            self._by_ticker[t].sort(key=lambda d: d["date"])

        # BM25 over all docs for the semantic fallback.
        logger.info("Building BM25 index over %d training docs …", len(corpus))
        self._tokens = [_tokenize(d["text"][:3000]) for d in corpus]
        self._bm25   = _BM25(self._tokens)
        logger.info("BM25 index ready.")

    @classmethod
    def from_pairs(cls, rows: list[dict], train_ratio: float = 0.9,
                   excerpt_chars: int = 2000) -> "TranscriptRetriever":
        """
        Build a retriever from the training portion of a JSONL pairs list.

        Parameters
        ----------
        rows         : full pairs list already sorted chronologically
                       (as returned by finetune.load_pairs)
        train_ratio  : fraction used for training — must match the split in
                       finetune.py / evaluate.py (default 0.9)
        excerpt_chars: how many characters of each transcript to index and
                       return.  Kept short to control token usage at inference.
        """
        n_train = int(len(rows) * train_ratio)
        corpus = [
            {
                "ticker": r["ticker"],
                "date":   r["date"][:10],
                "text":   _extract_transcript(r["input"])[:excerpt_chars],
            }
            for r in rows[:n_train]
        ]
        logger.info(
            "TranscriptRetriever: corpus=%d docs from %d training rows",
            len(corpus), n_train,
        )
        return cls(corpus)

    def retrieve(self, ticker: str, date: str, query_text: str,
                 k: int = 2) -> list[dict]:
        """
        Return up to k corpus documents relevant to the query.

        Priority order:
          1. Same ticker, most-recent prior dates (company-specific context)
          2. BM25-ranked docs from other tickers (general domain context)

        Parameters
        ----------
        ticker      : ticker symbol of the query transcript
        date        : date of the query transcript ("YYYY-MM-DD")
        query_text  : transcript text used to score BM25 candidates
        k           : maximum number of passages to return

        Returns
        -------
        list of dicts {ticker, date, text}
        """
        qdate = date[:10]

        # Step 1 — company-prior (most recent first)
        prior = sorted(
            [d for d in self._by_ticker.get(ticker, []) if d["date"] < qdate],
            key=lambda d: d["date"], reverse=True,
        )
        selected    = prior[:k]
        selected_ids = {id(d) for d in selected}

        # Step 2 — BM25 fallback from other tickers
        remaining = k - len(selected)
        if remaining > 0:
            q_tokens = _tokenize(query_text[:2000])
            for idx in self._bm25.top_n(q_tokens, n=len(self.corpus)):
                if remaining <= 0:
                    break
                doc = self.corpus[idx]
                if id(doc) not in selected_ids and doc["ticker"] != ticker:
                    selected.append(doc)
                    selected_ids.add(id(doc))
                    remaining -= 1

        return selected


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_rag_prefix(passages: list[dict],
                     max_chars_per_passage: int = 800) -> str:
    """
    Format retrieved passages as a context block to prepend to the prompt.

    Returns an empty string when no passages are provided so callers can
    safely do ``if rag_prefix:`` without special-casing.

    Layout::

        Retrieved context from prior earnings calls:

        --- AAPL | 2023-01-26 (Q4 2022) ---
        Revenue of $117.2 billion …

        --- AAPL | 2022-10-27 (Q3 2022) ---
        We generated revenue of $90.1 billion …
    """
    if not passages:
        return ""
    parts = ["Retrieved context from prior earnings calls:"]
    for p in passages:
        ticker  = p.get("ticker", "")
        date    = p.get("date",   "")
        text    = p.get("text",   "").strip()
        qlabel  = _quarter_label(date)
        excerpt = text[:max_chars_per_passage]
        if len(text) > max_chars_per_passage:
            excerpt += " …"
        parts.append(f"\n--- {ticker} | {date} ({qlabel}) ---\n{excerpt}")
    return "\n".join(parts)
