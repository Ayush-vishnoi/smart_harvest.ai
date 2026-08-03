"""Memory-safe retrieval for the Smart Harvest AI chatbot."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DOCS_DIR = Path(__file__).resolve().parents[1] / "farming_docs"
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "should", "the", "to", "what", "with",
}


def _tokens(value: str) -> set[str]:
    return {word for word in WORD_PATTERN.findall(value.lower()) if len(word) > 1 and word not in STOP_WORDS}


def _title(path: Path) -> str:
    return path.stem.replace("___", " - ").replace("_", " ").strip().title()


@lru_cache(maxsize=1)
def _documents() -> tuple[dict[str, Any], ...]:
    """Load the small checked-in knowledge base once without ML runtime dependencies."""
    documents: list[dict[str, Any]] = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            title = _title(path)
            documents.append({
                "text": text,
                "metadata": {"title": title, "source": path.name},
                "tokens": _tokens(f"{title} {text}"),
                "title_tokens": _tokens(title),
            })
    if not documents:
        raise RuntimeError("The local farming knowledge base is unavailable")
    return tuple(documents)


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return locally ranked farming passages without loading PyTorch in production."""
    clean_query = query.strip() if isinstance(query, str) else ""
    if not clean_query:
        raise ValueError("A non-empty query is required")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")

    query_tokens = _tokens(clean_query)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for document in _documents():
        overlap = query_tokens & document["tokens"]
        title_overlap = query_tokens & document["title_tokens"]
        score = float(len(overlap) + (3 * len(title_overlap)))
        if score:
            ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {"text": document["text"], "metadata": document["metadata"], "score": score}
        for score, document in ranked[:top_k]
    ]
