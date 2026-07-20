"""Text preprocessing placeholders."""

from __future__ import annotations

import re


def clean_visible_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def truncate(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
