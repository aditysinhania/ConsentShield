"""Checkpoint save / load utilities."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _torch():
    import torch

    return torch


def save_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    label_vocab: dict[str, int] | None = None,
    rng_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save model + training state.

    Always stores config, seed-related RNG state, and metadata alongside weights.
    """
    torch = _torch()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if rng_state is None:
        rng_state = capture_rng_state()

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "metrics": metrics or {},
        "config": config or {},
        "label_vocab": label_vocab or {},
        "rng_state": rng_state,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    if model is not None:
        payload["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    # Atomic-ish write: avoid corrupting the target if the process dies mid-save
    # (important on OneDrive / low-disk environments).
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta = {
        "path": str(out),
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "best_metric": best_metric,
        "metrics": metrics or {},
        "saved_at": payload["saved_at"],
        "has_model": model is not None,
        "has_optimizer": optimizer is not None,
        "has_scheduler": scheduler is not None,
        "label_vocab": label_vocab or {},
        "config": config or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    torch = _torch()
    return torch.load(Path(path), map_location=map_location or "cpu", weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    torch = _torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch = _torch()
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointCallback:
    """Save best + latest checkpoints each epoch."""

    def __init__(
        self,
        directory: str | Path,
        *,
        monitor: str = "val_f1",
        mode: str = "max",
        filename_best: str = "best.pt",
        filename_latest: str = "latest.pt",
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.filename_best = filename_best
        self.filename_latest = filename_latest
        self.best_metric: float | None = None

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        return value > self.best_metric if self.mode == "max" else value < self.best_metric

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        config: dict[str, Any] | None = None,
        label_vocab: dict[str, int] | None = None,
        global_step: int = 0,
    ) -> dict[str, Path]:
        latest = self.directory / self.filename_latest
        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=self.best_metric,
            metrics=metrics,
            config=config,
            label_vocab=label_vocab,
        )
        paths = {"latest": latest}

        if self.monitor in metrics and self._is_better(float(metrics[self.monitor])):
            self.best_metric = float(metrics[self.monitor])
            best = self.directory / self.filename_best
            save_checkpoint(
                best,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=self.best_metric,
                metrics=metrics,
                config=config,
                label_vocab=label_vocab,
            )
            paths["best"] = best
        return paths
