# Dense retriever backed by a pre-built FAISS IndexFlatIP
# Loads index.faiss + metadata.jsonl, retrieves passages at query time,
# and formats them into a context string for prompt injection

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embed_utils import (
    extract_transcript_for_embedding,
    validate_faiss_index,
    EXPECTED_DIM,
)

logger = logging.getLogger(__name__)

# Dense retriever: call load() once, then retrieve() + format_context() per query
class Retriever:

    def __init__(
        self,
        index_dir: str,
        embed_model,
        bge_tokenizer,
        rag_k: int = 3,
        rag_token_budget: int = 400,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embed_model = embed_model
        self.bge_tokenizer = bge_tokenizer
        self.rag_k = rag_k
        self.rag_token_budget = rag_token_budget
        self.index = None
        self.metadata: List[Dict] = []

    # Load index.faiss and metadata.jsonl
    # run all four validation checks
    def load(self) -> None:
        import faiss

        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.jsonl"

        if not index_path.exists():
            raise RuntimeError(
                f"index.faiss not found at {index_path}. "
                "Run build_index.py first (scripts/build_index.sh)."
            )
        if not meta_path.exists():
            raise RuntimeError(
                f"metadata.jsonl not found at {meta_path}. "
                "Run build_index.py first (scripts/build_index.sh)."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        validate_faiss_index(self.index, self.metadata)
        logger.info(
            "Retriever ready: index_dir=%s  passages=%d  rag_k=%d",
            self.index_dir, len(self.metadata), self.rag_k,
        )

    # Retrieve up to rag_k passages: same-company heuristic fills slot 0, FAISS fills the rest
    # embedding_cache is keyed by (ticker, date) to avoid re-encoding the same transcript
    def retrieve(self, row: dict, embedding_cache: dict) -> List[Dict]:
        assert self.index is not None, "Call retriever.load() before retrieve()"

        # Embed query, using cache to skip re-encoding the same transcript
        cache_key = (row["ticker"], row["date"])
        if cache_key in embedding_cache:
            q_vec = embedding_cache[cache_key]
        else:
            q_vec = extract_transcript_for_embedding(
                row, self.bge_tokenizer, self.embed_model
            )
            embedding_cache[cache_key] = q_vec

        # Same-company heuristic: most recent prior call from the same ticker
        # ISO date strings sort correctly as plain strings (YYYY-MM-DD)
        heuristic_candidates = [
            m for m in self.metadata
            if m["ticker"] == row["ticker"] and m["date"] < row["date"]
        ]
        heuristic_candidates.sort(key=lambda m: m["date"], reverse=True)
        heuristic_results = heuristic_candidates[:1]

        # Dense FAISS search — over-retrieve to have room after deduplication
        search_k = self.rag_k + len(heuristic_results) + 2
        scores, faiss_ids = self.index.search(
            q_vec.reshape(1, -1).astype(np.float32), search_k
        )
        # faiss_id == -1 when the index has fewer vectors than search_k
        dense_results = [
            self.metadata[fid]
            for fid in faiss_ids[0]
            if 0 <= fid < len(self.metadata)
        ]

        # Merge: heuristic first, then dense, deduplicating by stable integer id
        seen_ids: set = set()
        merged: List[Dict] = []

        for m in heuristic_results:
            if m["id"] not in seen_ids and len(merged) < self.rag_k:
                merged.append(m)
                seen_ids.add(m["id"])

        for m in dense_results:
            if m["id"] not in seen_ids and len(merged) < self.rag_k:
                merged.append(m)
                seen_ids.add(m["id"])

        return merged

    # Format retrieved passages into a context block; splits the token budget evenly across passages
    # Returns (context_string, actual_qwen_token_count); returns ("", 0) if passages is empty
    def format_context(
        self,
        passages: List[Dict],
        qwen_tokenizer,
        total_token_budget: int,
    ) -> Tuple[str, int]:
        if not passages:
            return "", 0

        n = len(passages)
        budget_per_passage = total_token_budget // n
        remainder = total_token_budget - budget_per_passage * n

        parts = []
        for i, passage in enumerate(passages):
            p_budget = budget_per_passage + (remainder if i == 0 else 0)
            text = passage["text_excerpt"]
            ids = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > p_budget:
                ids = ids[:p_budget]
                text = qwen_tokenizer.decode(ids, skip_special_tokens=True)
            header = f"[{passage['ticker']} | {passage['date']}]"
            parts.append(f"{header}\n{text}")

        context_str = "\n\n".join(parts)
        actual_tokens = len(
            qwen_tokenizer(context_str, add_special_tokens=False)["input_ids"]
        )
        return context_str, actual_tokens
