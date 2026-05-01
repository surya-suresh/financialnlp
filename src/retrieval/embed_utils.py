# Shared embedding utilities for the retrieval pipeline
# Handles transcript extraction, BGE tokenizer truncation, and vector encoding
# Used by both build_index.py (offline indexing) and retriever.py (query time)
# Also contains FAISS index validation called from both places

import logging
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Delimiter set by build_pairs.py — must stay in sync
DELIMITER = "\n\nTranscript:\n"

# BGE model hard limits
BGE_MAX_SEQ_LENGTH = 512
EXPECTED_DIM = 384

# Load the BGE embedding model and its tokenizer from a local path
def load_embed_model(model_path: str) -> Tuple:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    logger.info("Loading embedding model from %s", model_path)
    embed_model = SentenceTransformer(model_path)
    embed_model.max_seq_length = BGE_MAX_SEQ_LENGTH
    bge_tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Embedding model loaded (dim=%d, max_seq=%d)", EXPECTED_DIM, BGE_MAX_SEQ_LENGTH)
    return embed_model, bge_tokenizer

# Extract just the transcript text from a data row
# Falls back to the full input string if the delimiter is missing
def extract_raw_transcript_text(row: dict) -> str:
    parts = row["input"].split(DELIMITER, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    logger.warning(
        "Delimiter %r not found in row (ticker=%s, date=%s); using full input as fallback",
        DELIMITER, row.get("ticker", "?"), row.get("date", "?"),
    )
    return row["input"].strip()

# Truncate text to the BGE model's 512-token limit using the bge tokenizer
def truncate_text_for_embedding(raw_text: str, bge_tokenizer) -> str:
    token_ids = bge_tokenizer(
        raw_text,
        # [CLS]/[SEP] consume 2 of the 512 slots
        add_special_tokens=True,
        truncation=True,
        max_length=BGE_MAX_SEQ_LENGTH,
        return_tensors=None,
    )["input_ids"]
    return bge_tokenizer.decode(token_ids, skip_special_tokens=True)

# Extract, truncate, and embed a single row's transcript
# Single shared path used at both index build time and query time
def extract_transcript_for_embedding(
    row: dict,
    bge_tokenizer,
    embed_model,
) -> np.ndarray:
    raw_text = extract_raw_transcript_text(row)
    truncated_text = truncate_text_for_embedding(raw_text, bge_tokenizer)
    vector = embed_model.encode(
        truncated_text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector.astype(np.float32)

# Embed a list of pre-extracted transcript strings in batches
def batch_embed(
    texts: list,
    embed_model,
    batch_size: int = 64,
) -> np.ndarray:
    vectors = embed_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)

# Check that the FAISS index and metadata are consistent and non-empty
# Raises RuntimeError with a clear message on any failure
def validate_faiss_index(index, metadata: list) -> None:
    if index.ntotal == 0:
        raise RuntimeError(
            "FAISS index is empty (ntotal=0). "
            "Was build_index.py run successfully? Check the build job logs."
        )
    if len(metadata) != index.ntotal:
        raise RuntimeError(
            f"Metadata length ({len(metadata)}) does not match FAISS index size "
            f"({index.ntotal}). Index and metadata are out of sync — "
            "delete both files and rerun build_index.py."
        )
    if index.d != EXPECTED_DIM:
        raise RuntimeError(
            f"FAISS index dimension ({index.d}) does not match expected dimension "
            f"({EXPECTED_DIM}). The index was built with a different embedding model — "
            "delete and rebuild with bge-small-en-v1.5."
        )
    if metadata[0]["id"] != 0 or metadata[-1]["id"] != len(metadata) - 1:
        raise RuntimeError(
            f"Metadata ID field is not contiguous from 0 to {len(metadata) - 1}. "
            "metadata.jsonl may be truncated or corrupted — rebuild both artefacts."
        )
    logger.info(
        "FAISS index validation passed: ntotal=%d, dim=%d, metadata_lines=%d",
        index.ntotal, index.d, len(metadata),
    )
