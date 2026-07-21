"""Dependency-injection model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.registry.interfaces import EmbeddingEngine, ReasoningEngine, TextClassifier, VisionClassifier
from ai.registry.stubs import (
    StubEmbeddingEngine,
    StubReasoningEngine,
    TextClassifierAdapter,
    VisionClassifierAdapter,
)


@dataclass
class ModelRegistry:
    """Central registry for injectable model interfaces."""

    text: TextClassifier = field(default_factory=TextClassifierAdapter)
    vision: VisionClassifier = field(default_factory=VisionClassifierAdapter)
    embedding: EmbeddingEngine = field(default_factory=StubEmbeddingEngine)
    reasoning: ReasoningEngine = field(default_factory=StubReasoningEngine)

    @classmethod
    def default(cls) -> ModelRegistry:
        return cls()

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "interface": "TextClassifier",
                "name": self.text.name,
                "version": self.text.version,
                "ready": self.text.is_ready(),
            },
            {
                "interface": "VisionClassifier",
                "name": self.vision.name,
                "version": self.vision.version,
                "ready": self.vision.is_ready(),
            },
            {
                "interface": "EmbeddingEngine",
                "name": self.embedding.name,
                "version": self.embedding.version,
                "ready": self.embedding.is_ready(),
            },
            {
                "interface": "ReasoningEngine",
                "name": self.reasoning.name,
                "version": self.reasoning.version,
                "ready": self.reasoning.is_ready(),
            },
        ]
