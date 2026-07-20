"""Embedding helpers — placeholders for future sentence embeddings."""

from __future__ import annotations

from typing import Any


def embed_texts(texts: list[str]) -> dict[str, Any]:
    return {
        "status": "not_loaded",
        "embeddings": [],
        "message": "Embedding model not configured.",
    }
