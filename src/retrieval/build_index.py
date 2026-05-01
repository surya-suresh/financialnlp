# Offline FAISS index builder for dense RAG retrieval
# Run once per task: python -m src.retrieval.build_index --pairs ... --out-dir ... --model-path ...
# --train-ratio MUST match evaluate_rag.py (default 0.9) to exclude test rows from the index

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.embed_utils import (
    load_embed_model,
    extract_raw_transcript_text,
    truncate_text_for_embedding,
    batch_embed,
    validate_faiss_index,
    EXPECTED_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load and date-sort JSONL pairs - must match evaluate_rag.py:load_pairs ordering
def load_pairs(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("date", ""))
    return rows

def main() -> None:
    import faiss

    ap = argparse.ArgumentParser(description="Build FAISS index for RAG retrieval")
    ap.add_argument("--pairs", type=Path, required=True,
                    help="Path to JSONL pairs file (pairs_direction.jsonl or pairs_eps_surprise.jsonl)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory — will contain index.faiss and metadata.jsonl")
    ap.add_argument("--model-path", type=str, required=True,
                    help="Local path to BAAI/bge-small-en-v1.5 model directory")
    ap.add_argument("--train-ratio", type=float, default=0.9,
                    help="Training split fraction; must match evaluate_rag.py (default 0.9)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Embedding batch size for SentenceTransformer.encode()")
    args = ap.parse_args()

    logger.info(
        "build_index: pairs=%s  out=%s  train_ratio=%.2f",
        args.pairs, args.out_dir, args.train_ratio,
    )

    # Take the training split — int(n * train_ratio) matches chrono_test_split exactly
    rows = load_pairs(args.pairs)
    n = len(rows)
    k = int(n * args.train_ratio)
    train_rows = rows[:k]
    logger.info("Total pairs: %d | training split: %d | held-out: %d", n, k, n - k)

    if k == 0:
        raise RuntimeError("Training split is empty — check --pairs and --train-ratio.")

    embed_model, bge_tokenizer = load_embed_model(args.model_path)

    # Extract, truncate (512 BGE tokens), and collect text excerpts for each training row
    # truncated text strings passed to batch_embed
    texts: list = []
    # one dict per training row, written to metadata.jsonl
    metadata: list = []

    logger.info("Extracting and tokenizing %d training transcripts ...", k)
    for i, row in enumerate(train_rows):
        raw_text = extract_raw_transcript_text(row)
        truncated_text = truncate_text_for_embedding(raw_text, bge_tokenizer)
        texts.append(truncated_text)
        metadata.append({
            # stable integer ID == FAISS position
            "id": i,
            "ticker": row["ticker"],
            "date": row["date"],
            # stored for context formatting at query time
            "text_excerpt": truncated_text,
        })
        if (i + 1) % 1000 == 0:
            logger.info("  extracted %d / %d", i + 1, k)

    logger.info("Batch-embedding %d texts (batch_size=%d) ...", len(texts), args.batch_size)
    vectors = batch_embed(texts, embed_model, batch_size=args.batch_size)
    logger.info("Embeddings complete: shape=%s dtype=%s", vectors.shape, vectors.dtype)

    assert vectors.shape == (k, EXPECTED_DIM), (
        f"Unexpected embedding shape {vectors.shape}; expected ({k}, {EXPECTED_DIM})"
    )

    # IndexFlatIP with L2-normalised vectors: inner product == cosine similarity
    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(vectors)
    logger.info("FAISS index built: ntotal=%d", index.ntotal)

    # Save index and metadata; metadata[i]["id"] == i == FAISS position always
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.faiss"
    meta_path = args.out_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))
    logger.info("Saved index  → %s", index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info("Saved metadata → %s (%d records)", meta_path, len(metadata))

    validate_faiss_index(index, metadata)
    logger.info("Build complete.")

if __name__ == "__main__":
    main()
