"""Load fine-tuned MiniLM checkpoints for registry / offline use."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ai.training.datasets._helpers import REPO_ROOT


DEFAULT_CHECKPOINT_CANDIDATES = (
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
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
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
