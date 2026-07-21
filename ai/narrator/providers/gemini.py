"""Gemini narrator provider."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai.narrator.context import NARRATOR_SYSTEM_PROMPT


def generate_gemini_summary(
    *,
    api_key: str,
    context: dict[str, Any],
    model: str = "gemini-2.0-flash",
    timeout: float = 30.0,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": NARRATOR_SYSTEM_PROMPT},
                    {"text": json.dumps(context, indent=2, default=str)},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned empty summary")
    return text
