"""Optional LLM executive summary — rules remain source of truth."""

from ai.narrator.context import build_narrator_context
from ai.narrator.fallback import deterministic_summary
from ai.narrator.narrator import LLMNarrator, NarratorResult

__all__ = [
    "LLMNarrator",
    "NarratorResult",
    "build_narrator_context",
    "deterministic_summary",
]
