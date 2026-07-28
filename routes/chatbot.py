"""Flask routes for the RAG-powered farming assistant."""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from flask import Blueprint, jsonify, request

from utils.rag_helper import retrieve_relevant_chunks

chatbot_bp = Blueprint("chatbot", __name__)
log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = (
    "You are a helpful farming assistant for Smart Harvest AI. Answer using the provided "
    "context when relevant. If the context doesn't cover the question, say so honestly rather "
    "than guessing. For pesticide or chemical dosage questions, always recommend the user "
    "confirm exact amounts with a local agricultural extension officer rather than stating a "
    "specific number with full confidence. Keep answers concise and practical, since users are "
    "farmers looking for quick, clear guidance."
)
MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_MESSAGES = 12


def _clean_history(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("history must be an array")

    cleaned: list[dict[str, str]] = []
    for entry in value[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            cleaned.append({"role": role, "content": content[:MAX_MESSAGE_LENGTH]})
    return cleaned


def _source_titles(chunks: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        title = str(metadata.get("title") or metadata.get("disease_name") or "Farming knowledge base")
        if title not in sources:
            sources.append(title)
    return sources


@chatbot_bp.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A valid JSON object is required"}), 400

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "message is required"}), 400
    message = message.strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"message must be at most {MAX_MESSAGE_LENGTH} characters"}), 400

    try:
        history = _clean_history(payload.get("history"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return jsonify({"error": "Chat service is not configured"}), 503

    try:
        chunks = retrieve_relevant_chunks(message, top_k=3)
    except RuntimeError as exc:
        log.warning("RAG configuration error: %s", exc)
        return jsonify({"error": "Knowledge retrieval is not configured"}), 503
    except Exception:
        log.exception("Pinecone retrieval failed")
        return jsonify({"error": "Knowledge retrieval is temporarily unavailable"}), 502

    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    context_message = (
        "Retrieved farming knowledge:\n"
        f"{context if context else 'No relevant knowledge-base passages were found.'}\n\n"
        "Use this context when relevant, and clearly acknowledge when it does not answer the question."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_message},
        *history,
        {"role": "user", "content": message},
    ]

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 500},
            timeout=(5, 30),
        )
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        if not reply:
            raise ValueError("Groq returned an empty reply")
    except requests.Timeout:
        log.warning("Groq request timed out")
        return jsonify({"error": "The farming assistant timed out. Please try again."}), 504
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        log.warning("Groq API request failed (status=%s): %s", status, exc)
        return jsonify({"error": "The farming assistant is temporarily unavailable"}), 502
    except (KeyError, IndexError, TypeError, ValueError):
        log.exception("Groq returned an invalid response")
        return jsonify({"error": "The farming assistant returned an invalid response"}), 502

    return jsonify({"reply": reply, "sources": _source_titles(chunks)})
