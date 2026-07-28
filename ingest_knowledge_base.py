#!/usr/bin/env python3
"""Embed and upload Smart Harvest AI farming knowledge to Pinecone.

Run this script locally once after creating a Pinecone index with dimension 384
and metric cosine. It is intentionally not imported by the Flask application.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Keep transformers from probing TensorFlow/Keras 3 when loading PyTorch embeddings.
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

try:
    from pinecone import Pinecone
except ImportError as exc:  # pragma: no cover - exercised when dependencies are absent
    raise SystemExit("Install the dependencies first: python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100


def load_json(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object at {path}")
    return value


def farming_chunk(key: str, item: dict[str, Any]) -> tuple[str, dict[str, str]]:
    title = str(item.get("title") or key.replace("_", " ").title())
    text = "\n".join(
        [
            f"Title: {title}",
            f"Category: {item.get('category', 'General farming')}",
            f"Content: {item.get('content', '')}",
            f"Relevance to Yield: {item.get('relevance_to_yield', '')}",
        ]
    )
    metadata = {
        "source_type": "farming_topic",
        "title": title,
        "category": str(item.get("category", "General farming")),
        "text": text,
    }
    return text, metadata


def disease_chunk(key: str, item: dict[str, Any]) -> tuple[str, dict[str, str]]:
    disease_name = str(item.get("disease_name") or key.replace("_", " "))
    text = "\n".join(
        [
            f"Crop: {item.get('crop', '')}",
            f"Disease Name: {disease_name}",
            f"Pathogen: {item.get('pathogen', '')}",
            f"Symptoms: {item.get('symptoms', '')}",
            f"Favorable Conditions: {item.get('favorable_conditions', '')}",
            f"Organic Treatment: {item.get('organic_treatment', '')}",
            f"Chemical Treatment: {item.get('chemical_treatment', '')}",
            f"Prevention: {item.get('prevention', '')}",
        ]
    )
    metadata = {
        "source_type": "disease",
        "title": f"{item.get('crop', '')} - {disease_name}",
        "disease_name": disease_name,
        "category": "Plant disease",
        "text": text,
    }
    return text, metadata


def build_documents() -> list[tuple[str, str, dict[str, str]]]:
    topics = load_json(ROOT / "farming_docs" / "all_farming_topics.json")
    diseases = load_json(ROOT / "farming_docs" / "all_diseases.json")
    documents: list[tuple[str, str, dict[str, str]]] = []
    for key, item in topics.items():
        text, metadata = farming_chunk(key, item)
        documents.append((f"topic-{key}", text, metadata))
    for key, item in diseases.items():
        text, metadata = disease_chunk(key, item)
        documents.append((f"disease-{key}", text, metadata))
    return documents


def main() -> int:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        print("Set PINECONE_API_KEY and PINECONE_INDEX_NAME before running ingestion.", file=sys.stderr)
        return 1

    documents = build_documents()
    print(f"Loaded {len(documents)} knowledge entries.")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [text for _, text, _ in documents],
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    embedded = 0
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch = documents[start : start + UPSERT_BATCH_SIZE]
        vectors = [
            {"id": doc_id, "values": embeddings[start + offset].tolist(), "metadata": metadata}
            for offset, (doc_id, _, metadata) in enumerate(batch)
        ]
        index.upsert(vectors=vectors)
        embedded += len(batch)
        print(f"{embedded}/{len(documents)} embedded and uploaded")

    topic_count = sum(1 for doc_id, _, _ in documents if doc_id.startswith("topic-"))
    disease_count = len(documents) - topic_count
    print("Ingestion complete.")
    print(f"Total vectors: {embedded}")
    print(f"Farming topics: {topic_count}")
    print(f"Diseases: {disease_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
