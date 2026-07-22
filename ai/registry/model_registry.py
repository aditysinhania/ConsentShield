"""Dependency-injection model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.models.config import Phase4ModelConfig
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
    config: Phase4ModelConfig | None = None

    @classmethod
    def default(cls) -> ModelRegistry:
        return cls.from_config(Phase4ModelConfig.from_env())

    @classmethod
    def from_config(cls, config: Phase4ModelConfig) -> ModelRegistry:
        """
        Build registry from config.
        AI_STUB_MODE=true → stubs (Phase 3.5 behavior).
        Otherwise → Phase 4 text/vision/embedding implementations (pretrained or lexical).
        """
        if config.stub_mode and config.backend == "auto":
            return cls(
                text=TextClassifierAdapter(),
                vision=VisionClassifierAdapter(),
                embedding=StubEmbeddingEngine(),
                reasoning=StubReasoningEngine(),
                config=config,
            )

        from ai.text.classifier.text_classifier import TextClassifier as TextClassifierImpl
        from ai.text.embeddings.encoder import SentenceTransformerEngine
        from ai.vision.detectors.vision_detector import VisionDetector

        embedding = SentenceTransformerEngine(config)
        text = TextClassifierAdapter(TextClassifierImpl(config=config, embedding=embedding))
        vision = VisionClassifierAdapter(VisionDetector(config=config))
        return cls(
            text=text,
            vision=vision,
            embedding=embedding,
            reasoning=StubReasoningEngine(),
            config=config,
        )

    def describe(self) -> list[dict[str, Any]]:
        cfg = self.config or Phase4ModelConfig.from_env()
        rows = [
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
        for row in rows:
            row["phase4"] = {
                "stub_mode": cfg.stub_mode,
                "backend": cfg.backend,
                "text_model": cfg.text_model_name,
                "vision_model": cfg.vision_model_name,
                "device": cfg.device,
            }
        return rows
