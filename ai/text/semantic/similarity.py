"""Similarity helpers shared by lexical and embedding backends."""

from __future__ import annotations

import math
import re
from typing import Sequence


_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def collect_candidate_texts(payload) -> list[str]:
    """Extract short consent-relevant snippets from a ScanPayload."""
    texts: list[str] = []
    css = payload.css_snapshot or {}
    for btn in css.get("buttons") or []:
        if isinstance(btn, dict):
            label = str(btn.get("text") or btn.get("ariaLabel") or "").strip()
            if label:
                texts.append(label)
    for box in list(css.get("checkboxes") or []) + list(css.get("toggles") or []):
        if isinstance(box, dict):
            label = str(box.get("label") or box.get("name") or "").strip()
            if label:
                texts.append(label)
    visible = (payload.visible_text or "").strip()
    if visible:
        # Keep first chunk + any sentence with cookie/consent tokens
        texts.append(visible[:400])
        for part in re.split(r"[.!?]\s+", visible):
            low = part.lower()
            if any(tok in low for tok in ("cookie", "consent", "privacy", "accept", "reject", "agree")):
                texts.append(part.strip()[:240])
    # Dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:40]
