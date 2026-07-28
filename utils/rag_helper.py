"""Pinecone retrieval for the Smart Harvest AI chatbot."""
from __future__ import annotations

import os
import threading
from typing import Any

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_init_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                # Keep transformers from probing TensorFlow/Keras 3 when the
                # app loads PyTorch sentence embeddings at runtime. Force CPU
                # on macOS because MPS compiler failures abort the process and
                # make Gunicorn repeatedly restart its worker.
                os.environ.setdefault("USE_TF", "0")
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _get_index():
    global _index
    if _index is None:
        with _init_lock:
            if _index is None:
                api_key = os.environ.get("PINECONE_API_KEY")
                index_name = os.environ.get("PINECONE_INDEX_NAME")
                if not api_key or not index_name:
                    raise RuntimeError("Pinecone is not configured")
                from pinecone import Pinecone

                _index = Pinecone(api_key=api_key).Index(index_name)
    return _index


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Embed a query and return the best matching Pinecone chunks."""
    clean_query = query.strip() if isinstance(query, str) else ""
    if not clean_query:
        raise ValueError("A non-empty query is required")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")

    vector = _get_model().encode(clean_query, normalize_embeddings=True).tolist()
    result = _get_index().query(vector=vector, top_k=top_k, include_metadata=True)
    matches = _field(result, "matches", []) or []
    chunks: list[dict[str, Any]] = []
    for match in matches:
        metadata = dict(_field(match, "metadata", {}) or {})
        text = str(metadata.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            {
                "text": text,
                "metadata": metadata,
                "score": float(_field(match, "score", 0.0) or 0.0),
            }
        )
    return chunks
