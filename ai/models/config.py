"""Phase 4 model runtime settings (lazy; safe defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase4ModelConfig:
    models_root: Path
    text_model_name: str
    vision_model_name: str
    device: str
    embedding_batch_size: int
    cache_models: bool
    stub_mode: bool
    backend: str  # auto | pretrained | lexical

    @classmethod
    def from_env(cls) -> Phase4ModelConfig:
        # Prefer FastAPI settings when available (loads .env)
        try:
            from app.core.config import settings

            stub = bool(settings.AI_STUB_MODE)
            backend = (settings.PHASE4_BACKEND or "auto").strip().lower()
            device = (settings.DEVICE or "cpu").strip().lower()
            if device not in ("cpu", "cuda"):
                device = "cpu"
            text_name = settings.TEXT_MODEL_NAME or settings.TEXT_MODEL_PATH or (
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            vision_name = settings.VISION_MODEL_NAME or settings.VISION_MODEL_PATH or (
                "openai/clip-vit-base-patch32"
            )
            return cls(
                models_root=Path(settings.MODELS_ROOT).expanduser(),
                text_model_name=text_name,
                vision_model_name=vision_name,
                device=device,
                embedding_batch_size=max(1, int(settings.EMBEDDING_BATCH_SIZE)),
                cache_models=bool(settings.CACHE_MODELS),
                stub_mode=stub,
                backend=backend if backend in ("auto", "pretrained", "lexical") else "auto",
            )
        except Exception:
            pass

        root = Path(os.getenv("MODELS_ROOT", "./models")).expanduser()
        stub = os.getenv("AI_STUB_MODE", "true").lower() in ("1", "true", "yes", "on")
        backend = os.getenv("PHASE4_BACKEND", "auto").strip().lower()
        device = os.getenv("DEVICE", "cpu").strip().lower()
        if device not in ("cpu", "cuda"):
            device = "cpu"
        return cls(
            models_root=root,
            text_model_name=os.getenv(
                "TEXT_MODEL_NAME",
                os.getenv("TEXT_MODEL_PATH") or "sentence-transformers/all-MiniLM-L6-v2",
            ),
            vision_model_name=os.getenv(
                "VISION_MODEL_NAME",
                os.getenv("VISION_MODEL_PATH") or "openai/clip-vit-base-patch32",
            ),
            device=device,
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))),
            cache_models=os.getenv("CACHE_MODELS", "true").lower() in ("1", "true", "yes", "on"),
            stub_mode=stub,
            backend=backend if backend in ("auto", "pretrained", "lexical") else "auto",
        )

    @property
    def cache_dir(self) -> Path:
        return self.models_root / "cache"
