"""OpenAI narrator provider."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai.narrator.context import NARRATOR_SYSTEM_PROMPT


def generate_openai_summary(
    *,
    api_key: str,
    context: dict[str, Any],
    model: str = "gpt-4o-mini",
    timeout: float = 30.0,
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": NARRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, indent=2, default=str)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices")
    text = (choices[0].get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError("OpenAI returned empty summary")
    return text
