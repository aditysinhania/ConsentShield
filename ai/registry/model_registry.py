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
    finetuned_minilm: Any | None = None
    finetuned_minilm_meta: dict[str, Any] | None = None
    _finetuned_bundle: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def default(cls) -> ModelRegistry:
        return cls.from_config(Phase4ModelConfig.from_env())

    @classmethod
    def from_config(cls, config: Phase4ModelConfig) -> ModelRegistry:
        """
        Build registry from config.

        AI_STUB_MODE=true + PHASE4_BACKEND=auto → vision/embedding stubs (Phase 3.5).
        Otherwise → Phase 4 text/vision/embedding implementations.

        Fine-tuned MiniLM is preferred for text when ``MINILM_CHECKPOINT`` (or the
        default checkpoint path) is available; exemplar similarity is the fallback.
        """
        from ai.text.classifier.text_classifier import TextClassifier as TextClassifierImpl

        if config.stub_mode and config.backend == "auto":
            text_impl = TextClassifierImpl(config=config, skip_finetuned_autoload=True)
            registry = cls(
                text=TextClassifierAdapter(text_impl),
                vision=VisionClassifierAdapter(),
                embedding=StubEmbeddingEngine(),
                reasoning=StubReasoningEngine(),
                config=config,
            )
        else:
            from ai.text.embeddings.encoder import SentenceTransformerEngine
            from ai.vision.detectors.vision_detector import VisionDetector

            embedding = SentenceTransformerEngine(config)
            text_impl = TextClassifierImpl(
                config=config,
                embedding=embedding,
                skip_finetuned_autoload=True,
            )
            vision = VisionClassifierAdapter(VisionDetector(config=config))
            registry = cls(
                text=TextClassifierAdapter(text_impl),
                vision=vision,
                embedding=embedding,
                reasoning=StubReasoningEngine(),
                config=config,
            )

        registry.load_finetuned_minilm(
            checkpoint=config.minilm_checkpoint,
            device=config.device,
        )
        return registry

    def load_finetuned_minilm(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """
        Load tokenizer + fine-tuned MiniLM classifier (singleton-cached).

        Returns a status dict: loaded, checkpoint, model_version, inference_device.
        On failure, status.loaded is False and text falls back to exemplars/stub.
        """
        from ai.text.classifier.finetuned_minilm import load_finetuned_minilm as _load

        dev = device or (self.config.device if self.config else "cpu")
        try:
            bundle = _load(checkpoint, device=dev, force_reload=force_reload)
        except Exception as exc:  # noqa: BLE001
            status = {
                "loaded": False,
                "status": "not_loaded",
                "checkpoint": str(checkpoint) if checkpoint else None,
                "model_version": None,
                "inference_device": dev,
                "name": "Fine-tuned MiniLM",
                "error": str(exc),
            }
            self.finetuned_minilm = None
            self.finetuned_minilm_meta = status
            self._finetuned_bundle = None
            self._bind_finetuned_to_text(None)
            return status

        status = {
            "loaded": True,
            "status": "loaded",
            "checkpoint": bundle.get("checkpoint"),
            "model_version": bundle.get("model_version"),
            "inference_device": bundle.get("inference_device") or bundle.get("device"),
            "name": "Fine-tuned MiniLM",
        }
        self.finetuned_minilm = bundle.get("model")
        self.finetuned_minilm_meta = {**status, "meta": bundle.get("meta")}
        self._finetuned_bundle = bundle
        self._bind_finetuned_to_text(bundle)
        return status

    def try_load_finetuned_minilm(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> bool:
        """Backward-compatible bool wrapper around ``load_finetuned_minilm``."""
        return bool(self.load_finetuned_minilm(checkpoint, device=device).get("loaded"))

    def _bind_finetuned_to_text(self, bundle: dict[str, Any] | None) -> None:
        inner = getattr(self.text, "_inner", None)
        if inner is not None and hasattr(inner, "bind_finetuned"):
            inner.bind_finetuned(bundle)

    def finetuned_status(self) -> dict[str, Any]:
        if self.finetuned_minilm_meta:
            return {
                "loaded": bool(self.finetuned_minilm_meta.get("loaded")),
                "status": self.finetuned_minilm_meta.get("status")
                or ("loaded" if self.finetuned_minilm else "not_loaded"),
                "checkpoint": self.finetuned_minilm_meta.get("checkpoint"),
                "model_version": self.finetuned_minilm_meta.get("model_version"),
                "inference_device": self.finetuned_minilm_meta.get("inference_device"),
                "name": "Fine-tuned MiniLM",
                "error": self.finetuned_minilm_meta.get("error"),
            }
        return {
            "loaded": False,
            "status": "not_loaded",
            "checkpoint": None,
            "model_version": None,
            "inference_device": self.config.device if self.config else "cpu",
            "name": "Fine-tuned MiniLM",
        }

    def describe(self) -> list[dict[str, Any]]:
        cfg = self.config or Phase4ModelConfig.from_env()
        ft = self.finetuned_status()
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
                "name": "Fine-tuned MiniLM",
                "version": ft.get("model_version") or "phase2",
                "ready": bool(ft.get("loaded")),
                "checkpoint": ft.get("checkpoint"),
                "inference_device": ft.get("inference_device"),
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
