"""Vision trainer scaffold for CLIP fine-tuning (Phase 3)."""

from __future__ import annotations

from typing import Any

from ai.training.trainers.base_trainer import BaseTrainer, PhaseGateError


class VisionTrainer(BaseTrainer):
    """
    CLIP / vision-language trainer.

    Phase 1: infrastructure only — ``build_model`` is intentionally unimplemented.
    Phase 3 will load ``openai/clip-vit-base-patch32`` (or configured backbone)
    and train with image (+ optional text) batches from ImageDataset.
    """

    task_name = "vision"

    def build_model(self, num_labels: int) -> Any:
        raise PhaseGateError(
            "VisionTrainer.build_model is reserved for Phase 3 (CLIP fine-tuning). "
            f"num_labels={num_labels}. Do not call fit() until Phase 3 is approved."
        )

    def training_step(self, batch: Any) -> tuple[Any, float]:
        raise PhaseGateError("VisionTrainer.training_step requires Phase 3 model implementation.")

    def predict_batch(self, batch: Any) -> tuple[list[Any], list[Any], list[str]]:
        raise PhaseGateError("VisionTrainer.predict_batch requires Phase 3 model implementation.")
