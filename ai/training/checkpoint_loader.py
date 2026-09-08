"""Load fine-tuned MiniLM checkpoints for registry / offline use."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ai.training.datasets._helpers import REPO_ROOT


DEFAULT_CHECKPOINT_CANDIDATES = (
    Path("models/checkpoints/minilm/best_model.pt"),
    Path("ai/training/runs/minilm/models/best_model.pt"),
    Path("ai/training/runs/minilm/models/best.pt"),
    Path("runs/minilm/models/best_model.pt"),
    Path("runs/minilm/models/best.pt"),
    Path("models/text/best_model.pt"),
)


def resolve_minilm_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """
    Resolve a fine-tuned MiniLM checkpoint path.

    Order: explicit arg → MINILM_CHECKPOINT env → default candidates.
    Returns None if nothing exists (inference keeps pretrained/stub behavior).
    When ``explicit`` is set, only that path is tried (no silent fallback).
    """
    if explicit is not None:
        raw = str(explicit).strip()
        if raw.lower() in ("", "none", "disabled", "__disabled__"):
            return None
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p if p.is_file() else None

    candidates: list[Path] = []
    env = os.getenv("MINILM_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    for rel in DEFAULT_CHECKPOINT_CANDIDATES:
        candidates.append(REPO_ROOT / rel if not rel.is_absolute() else rel)

    for path in candidates:
        p = path.expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.is_file():
            return p
    return None


def load_minilm_classifier(
    checkpoint: str | Path | None = None,
    *,
    device: str = "cpu",
    map_location: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Load ``MiniLMClassifier`` + metadata from a Phase 2 checkpoint.

    Returns ``(model.eval(), meta)`` where meta includes label_vocab and config.
    """
    from ai.training.callbacks.checkpoint import load_checkpoint
    from ai.training.models.minilm_classifier import MiniLMClassifier

    path = resolve_minilm_checkpoint(checkpoint)
    if path is None:
        raise FileNotFoundError(
            "No MiniLM fine-tuned checkpoint found. "
            "Train with `python -m ai.training.train_minilm` or set MINILM_CHECKPOINT."
        )

    ckpt = load_checkpoint(path, map_location=map_location or device)
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model") or {}
    label_vocab: dict[str, int] = dict(ckpt.get("label_vocab") or {})
    num_labels = len(label_vocab) if label_vocab else int(model_cfg.get("num_labels", 2))
    model = MiniLMClassifier(
        str(model_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2")),
        num_labels=num_labels,
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        freeze_layers=int(model_cfg.get("freeze_layers", 0)),
    )
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = {
        "checkpoint": str(path),
        "label_vocab": label_vocab,
        "id2label": {i: lab for lab, i in label_vocab.items()},
        "config": cfg,
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics") or {},
    }
    return model, meta


DEFAULT_CLIP_CHECKPOINT_CANDIDATES = (
    Path("models/checkpoints/clip/best_model.pt"),
    Path("ai/training/runs/clip/models/best_model.pt"),
    Path("ai/training/runs/clip/models/best.pt"),
    Path("runs/clip/models/best_model.pt"),
)


def resolve_clip_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """Resolve fine-tuned CLIP checkpoint (explicit only when provided)."""
    if explicit is not None:
        raw = str(explicit).strip()
        if raw.lower() in ("", "none", "disabled", "__disabled__"):
            return None
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p if p.is_file() else None

    candidates: list[Path] = []
    env = os.getenv("CLIP_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    for rel in DEFAULT_CLIP_CHECKPOINT_CANDIDATES:
        candidates.append(REPO_ROOT / rel if not rel.is_absolute() else rel)

    for path in candidates:
        p = path.expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.is_file():
            return p
    return None


def load_clip_classifier(
    checkpoint: str | Path | None = None,
    *,
    device: str = "cpu",
    map_location: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load ``CLIPClassifier`` + metadata from a Phase 3 checkpoint."""
    from ai.training.callbacks.checkpoint import load_checkpoint
    from ai.training.models.clip_classifier import CLIPClassifier

    path = resolve_clip_checkpoint(checkpoint)
    if path is None:
        raise FileNotFoundError(
            "No CLIP fine-tuned checkpoint found. "
            "Train with `python -m ai.training.train_clip` or set CLIP_CHECKPOINT."
        )

    ckpt = load_checkpoint(path, map_location=map_location or device)
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model") or {}
    label_vocab: dict[str, int] = dict(ckpt.get("label_vocab") or {})
    num_labels = len(label_vocab) if label_vocab else int(model_cfg.get("num_labels", 5))
    model = CLIPClassifier(
        str(model_cfg.get("name", "openai/clip-vit-base-patch32")),
        num_labels=num_labels,
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_text_encoder=bool(model_cfg.get("freeze_text_encoder", True)),
        freeze_projection=bool(model_cfg.get("freeze_projection", True)),
    )
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = {
        "checkpoint": str(path),
        "label_vocab": label_vocab,
        "id2label": {i: lab for lab, i in label_vocab.items()},
        "config": cfg,
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics") or {},
        "model_version": f"finetuned-clip-epoch{ckpt.get('epoch', '?')}",
        "enabled": os.getenv("CLIP_FINETUNED_ENABLED", "false").lower()
        in ("1", "true", "yes", "on"),
    }
    return model, meta
