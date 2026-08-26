"""Trainer package."""

from __future__ import annotations

# Lightweight exports only — avoid importing torch at package import time.
from ai.training.trainers.utils import load_config, set_seed

__all__ = [
    "BaseTrainer",
    "PhaseGateError",
    "TextTrainer",
    "VisionTrainer",
    "load_config",
    "set_seed",
]


def __getattr__(name: str):
    if name in {"BaseTrainer", "PhaseGateError"}:
        from ai.training.trainers.base_trainer import BaseTrainer, PhaseGateError

        return {"BaseTrainer": BaseTrainer, "PhaseGateError": PhaseGateError}[name]
    if name == "TextTrainer":
        from ai.training.trainers.text_trainer import TextTrainer

        return TextTrainer
    if name == "VisionTrainer":
        from ai.training.trainers.vision_trainer import VisionTrainer

        return VisionTrainer
    raise AttributeError(name)
