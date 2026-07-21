"""Model interface contracts — dependency injection only, no loaded weights."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ai.common.types import ScanPayload, TextPrediction, VisionFeatures


@runtime_checkable
class TextClassifier(Protocol):
    name: str
    version: str

    def classify(self, payload: ScanPayload) -> TextPrediction: ...
    def is_ready(self) -> bool: ...


@runtime_checkable
class VisionClassifier(Protocol):
    name: str
    version: str

    def extract_features(self, payload: ScanPayload) -> VisionFeatures: ...
    def is_ready(self) -> bool: ...


@runtime_checkable
class EmbeddingEngine(Protocol):
    name: str
    version: str

    def embed(self, texts: list[str]) -> dict[str, Any]: ...
    def is_ready(self) -> bool: ...


@runtime_checkable
class ReasoningEngine(Protocol):
    name: str
    version: str

    def summarize(self, report: dict[str, Any], *, url: str = "") -> dict[str, Any]: ...
    def is_ready(self) -> bool: ...
