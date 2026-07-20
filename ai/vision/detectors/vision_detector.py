"""Vision detector stub — returns structured features, never invents dark patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.common.base import BaseDetector, ModelNotLoadedError
from ai.common.types import EvalMetrics, PredictionResult, ScanPayload, TrainResult, VisionFeatures


class VisionDetector(BaseDetector[ScanPayload, PredictionResult]):
    """
    Future: YOLOv11 / Grounding DINO for banners, buttons, checkboxes, popups.

    Until weights are loaded, predict() returns status=not_loaded — never fake UI detections.
    """

    name = "vision_detector"
    version = "0.1.0"

    def __init__(self) -> None:
        self._model_path: str | None = None
        self._loaded = False

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(
            status="not_implemented",
            message="Vision training pipeline not yet implemented. See docs/Training.md.",
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        if not self._loaded:
            features = VisionFeatures(
                status="not_loaded",
                message="Vision model not loaded. Place weights under models/vision/ and call load_model().",
            )
            return PredictionResult(status="not_loaded", data=features.model_dump(), message=features.message)
        raise ModelNotLoadedError("Vision runtime path not implemented yet.")

    def extract_features(self, input_data: ScanPayload) -> VisionFeatures:
        result = self.predict(input_data)
        return VisionFeatures.model_validate(result.data)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="Vision evaluation pending model integration.")

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vision model not found: {path}")
        self._model_path = str(p)
        # Weights present but inference runtime not wired yet
        self._loaded = False
        raise NotImplementedError(
            "Vision weights found but YOLOv11/Grounding DINO inference is not wired yet."
        )

    def save_model(self, path: str) -> None:
        raise NotImplementedError("Vision save_model is not implemented yet.")

    def is_ready(self) -> bool:
        return self._loaded
