"""LLM narrator — optional Gemini/OpenAI with deterministic fallback."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.narrator.context import build_narrator_context
from ai.narrator.fallback import deterministic_summary
from ai.narrator.providers.gemini import generate_gemini_summary
from ai.narrator.providers.openai import generate_openai_summary

ProviderName = Literal["gemini", "openai", "offline", "none", ""]


class NarratorResult(BaseModel):
    summary: str
    source: str  # llm:gemini | llm:openai | deterministic
    provider: str | None = None
    status: str = "ready"  # ready | fallback | offline
    message: str | None = None
    duration_ms: float = 0.0
    context: dict[str, Any] = Field(default_factory=dict)


class LLMNarrator:
    """Summarize rule-engine output; never invent findings."""

    def __init__(
        self,
        *,
        provider: ProviderName = "offline",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        gemini_model: str = "gemini-2.0-flash",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.provider = (provider or "offline").strip().lower()
        self.gemini_api_key = gemini_api_key.strip()
        self.openai_api_key = openai_api_key.strip()
        self.gemini_model = gemini_model
        self.openai_model = openai_model

    @classmethod
    def from_env(
        cls,
        *,
        provider: str = "",
        gemini_api_key: str = "",
        openai_api_key: str = "",
    ) -> LLMNarrator:
        return cls(
            provider=provider or "offline",  # type: ignore[arg-type]
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
        )

    def _resolved_provider(self) -> str | None:
        p = self.provider
        if p in ("", "none", "offline"):
            return None
        if p == "gemini" and self.gemini_api_key:
            return "gemini"
        if p == "openai" and self.openai_api_key:
            return "openai"
        # Auto-select when provider set but only one key present
        if p == "gemini" and not self.gemini_api_key:
            return None
        if p == "openai" and not self.openai_api_key:
            return None
        return None

    def narrate(self, report: dict[str, Any], *, url: str = "") -> NarratorResult:
        t0 = time.perf_counter()
        context = build_narrator_context(report)
        resolved = self._resolved_provider()

        if resolved is None:
            summary = deterministic_summary(context, url=url)
            return NarratorResult(
                summary=summary,
                source="deterministic",
                provider=None,
                status="offline" if self.provider in ("offline", "none", "") else "fallback",
                message="LLM provider unavailable; using deterministic summary.",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                context=context,
            )

        try:
            if resolved == "gemini":
                summary = generate_gemini_summary(
                    api_key=self.gemini_api_key,
                    context=context,
                    model=self.gemini_model,
                )
            else:
                summary = generate_openai_summary(
                    api_key=self.openai_api_key,
                    context=context,
                    model=self.openai_model,
                )
            return NarratorResult(
                summary=summary,
                source=f"llm:{resolved}",
                provider=resolved,
                status="ready",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                context=context,
            )
        except Exception as exc:
            summary = deterministic_summary(context, url=url)
            return NarratorResult(
                summary=summary,
                source="deterministic",
                provider=resolved,
                status="fallback",
                message=str(exc),
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                context=context,
            )
