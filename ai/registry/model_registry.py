"""Dependency-injection model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    # Optional Phase 2 fine-tuned MiniLM (does not replace inference ``text`` by default)
    finetuned_minilm: Any | None = None
    finetuned_minilm_meta: dict[str, Any] | None = None

    @classmethod
    def default(cls) -> ModelRegistry:
        return cls.from_config(Phase4ModelConfig.from_env())

    @classmethod
    def from_config(cls, config: Phase4ModelConfig) -> ModelRegistry:
        """
        Build registry from config.
        AI_STUB_MODE=true → stubs (Phase 3.5 behavior).
        Otherwise → Phase 4 text/vision/embedding implementations (pretrained or lexical).

        If a Phase 2 MiniLM checkpoint is present (MINILM_CHECKPOINT or default path),
        it is loaded onto ``finetuned_minilm`` without changing the active inference
        ``text`` classifier — inference keeps working with or without the checkpoint.
        """
        if config.stub_mode and config.backend == "auto":
            registry = cls(
                text=TextClassifierAdapter(),
                vision=VisionClassifierAdapter(),
                embedding=StubEmbeddingEngine(),
                reasoning=StubReasoningEngine(),
                config=config,
            )
        else:
            from ai.text.classifier.text_classifier import TextClassifier as TextClassifierImpl
            from ai.text.embeddings.encoder import SentenceTransformerEngine
            from ai.vision.detectors.vision_detector import VisionDetector

            embedding = SentenceTransformerEngine(config)
            text = TextClassifierAdapter(TextClassifierImpl(config=config, embedding=embedding))
            vision = VisionClassifierAdapter(VisionDetector(config=config))
            registry = cls(
                text=text,
                vision=vision,
                embedding=embedding,
                reasoning=StubReasoningEngine(),
                config=config,
            )

        registry.try_load_finetuned_minilm(device=config.device)
        return registry

    def try_load_finetuned_minilm(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> bool:
        """
        Attempt to load a Phase 2 MiniLM checkpoint.

        Returns True if loaded. Missing checkpoints are ignored so inference
        continues with the existing text path.
        """
        try:
            from ai.training.checkpoint_loader import load_minilm_classifier, resolve_minilm_checkpoint
        except Exception:
            return False

        path = resolve_minilm_checkpoint(checkpoint)
        if path is None:
            self.finetuned_minilm = None
            self.finetuned_minilm_meta = None
            return False

        dev = device or (self.config.device if self.config else "cpu")
        try:
            model, meta = load_minilm_classifier(path, device=dev)
        except Exception as exc:
            self.finetuned_minilm = None
            self.finetuned_minilm_meta = {"error": str(exc), "checkpoint": str(path)}
            return False

        self.finetuned_minilm = model
        self.finetuned_minilm_meta = meta
        return True

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
            {
                "interface": "FineTunedMiniLM",
                "name": "minilm_classifier",
                "version": "phase2",
                "ready": self.finetuned_minilm is not None,
                "checkpoint": (self.finetuned_minilm_meta or {}).get("checkpoint"),
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
