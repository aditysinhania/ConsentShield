"""Stub model implementations for DI registry."""

from __future__ import annotations

from typing import Any

from ai.common.types import ScanPayload
from ai.narrator.narrator import LLMNarrator
from ai.text.classifier.text_classifier import TextClassifier as TextClassifierImpl
from ai.vision.detectors.vision_detector import VisionDetector


class StubEmbeddingEngine:
    name = "embedding_engine"
    version = "0.1.0"

    def embed(self, texts: list[str]) -> dict[str, Any]:
        return {
            "status": "not_loaded",
            "embeddings": [],
            "message": "Embedding model not configured (stub).",
        }

    def is_ready(self) -> bool:
        return False


class StubReasoningEngine:
    """Wraps LLM narrator; deterministic when provider offline."""

    name = "reasoning_engine"
    version = "0.1.0"

    def __init__(self, narrator: LLMNarrator | None = None) -> None:
        self._narrator = narrator or LLMNarrator(provider="offline")

    def summarize(self, report: dict[str, Any], *, url: str = "") -> dict[str, Any]:
        return self._narrator.narrate(report, url=url).model_dump()

    def is_ready(self) -> bool:
        return self._narrator._resolved_provider() is not None  # noqa: SLF001


class VisionClassifierAdapter:
    """Adapter: VisionDetector → VisionClassifier protocol."""

    def __init__(self, detector: VisionDetector | None = None) -> None:
        self._inner = detector or VisionDetector()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    def extract_features(self, payload: ScanPayload):
        return self._inner.extract_features(payload)

    def is_ready(self) -> bool:
        return self._inner.is_ready()


class TextClassifierAdapter:
    """Adapter: TextClassifierImpl → TextClassifier protocol."""

    def __init__(self, classifier: TextClassifierImpl | None = None) -> None:
        self._inner = classifier or TextClassifierImpl()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    def classify(self, payload: ScanPayload):
        return self._inner.classify(payload)

    def is_ready(self) -> bool:
        return self._inner.is_ready()
