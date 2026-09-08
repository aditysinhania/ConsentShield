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
    finetuned_clip: Any | None = None
    finetuned_clip_meta: dict[str, Any] | None = None
    _finetuned_clip_bundle: dict[str, Any] | None = field(default=None, repr=False)

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
        # Fine-tuned CLIP is opt-in (CLIP_FINETUNED_ENABLED) after Phase 3 eval passes.
        registry.load_finetuned_clip(
            checkpoint=config.clip_checkpoint,
            device=config.device,
            enabled=bool(config.clip_finetuned_enabled),
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

    def load_finetuned_clip(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """
        Load fine-tuned CLIP classifier for registry status / optional inference.

        Priority for vision inference (when enabled):
          Fine-tuned CLIP → Pretrained CLIP → Lexical fallback.

        Disabled by default (``CLIP_FINETUNED_ENABLED=false``) until Phase 3
        evaluation passes.
        """
        from ai.training.checkpoint_loader import resolve_clip_checkpoint

        cfg = self.config
        use = bool(cfg.clip_finetuned_enabled) if enabled is None and cfg else bool(enabled)
        path = resolve_clip_checkpoint(
            checkpoint if checkpoint is not None else (cfg.clip_checkpoint if cfg else None)
        )
        dev = device or (cfg.device if cfg else "cpu")

        if path is None:
            status = {
                "loaded": False,
                "status": "not_loaded",
                "enabled": use,
                "checkpoint": str(checkpoint) if checkpoint else (cfg.clip_checkpoint if cfg else None),
                "model_version": None,
                "inference_device": dev,
                "name": "Fine-tuned CLIP",
                "priority": ["finetuned_clip", "pretrained_clip", "lexical"],
            }
            self.finetuned_clip = None
            self.finetuned_clip_meta = status
            self._finetuned_clip_bundle = None
            return status

        if not use:
            status = {
                "loaded": False,
                "status": "available_disabled",
                "enabled": False,
                "checkpoint": str(path),
                "model_version": None,
                "inference_device": dev,
                "name": "Fine-tuned CLIP",
                "note": "Set CLIP_FINETUNED_ENABLED=true to activate after eval.",
                "priority": ["finetuned_clip", "pretrained_clip", "lexical"],
            }
            self.finetuned_clip = None
            self.finetuned_clip_meta = status
            self._finetuned_clip_bundle = None
            return status

        try:
            from ai.training.checkpoint_loader import load_clip_classifier

            model, meta = load_clip_classifier(path, device=dev)
        except Exception as exc:  # noqa: BLE001
            status = {
                "loaded": False,
                "status": "error",
                "enabled": True,
                "checkpoint": str(path),
                "model_version": None,
                "inference_device": dev,
                "name": "Fine-tuned CLIP",
                "error": str(exc),
                "priority": ["finetuned_clip", "pretrained_clip", "lexical"],
            }
            self.finetuned_clip = None
            self.finetuned_clip_meta = status
            self._finetuned_clip_bundle = None
            return status

        status = {
            "loaded": True,
            "status": "loaded",
            "enabled": True,
            "checkpoint": str(path),
            "model_version": meta.get("model_version"),
            "inference_device": dev,
            "name": "Fine-tuned CLIP",
            "label_vocab": meta.get("label_vocab"),
            "priority": ["finetuned_clip", "pretrained_clip", "lexical"],
        }
        self.finetuned_clip = model
        self.finetuned_clip_meta = {**status, "meta": meta}
        self._finetuned_clip_bundle = {"model": model, "meta": meta, **status}
        return status

    def finetuned_clip_status(self) -> dict[str, Any]:
        if self.finetuned_clip_meta:
            return {
                "loaded": bool(self.finetuned_clip_meta.get("loaded")),
                "status": self.finetuned_clip_meta.get("status") or "not_loaded",
                "enabled": bool(self.finetuned_clip_meta.get("enabled")),
                "checkpoint": self.finetuned_clip_meta.get("checkpoint"),
                "model_version": self.finetuned_clip_meta.get("model_version"),
                "inference_device": self.finetuned_clip_meta.get("inference_device"),
                "name": "Fine-tuned CLIP",
                "error": self.finetuned_clip_meta.get("error"),
                "note": self.finetuned_clip_meta.get("note"),
                "priority": self.finetuned_clip_meta.get("priority"),
            }
        return {
            "loaded": False,
            "status": "not_loaded",
            "enabled": False,
            "checkpoint": self.config.clip_checkpoint if self.config else None,
            "name": "Fine-tuned CLIP",
        }

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
        clip = self.finetuned_clip_status()
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
            {
                "interface": "FineTunedCLIP",
                "name": "Fine-tuned CLIP",
                "version": clip.get("model_version") or "phase3",
                "ready": bool(clip.get("loaded")),
                "enabled": bool(clip.get("enabled")),
                "status": clip.get("status"),
                "checkpoint": clip.get("checkpoint"),
                "inference_device": clip.get("inference_device"),
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
