"""Text classifier stub — no fabricated dark-pattern labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.common.base import BaseDetector, ModelNotLoadedError
from ai.common.types import EvalMetrics, PredictionResult, ScanPayload, TextPrediction, TrainResult


class TextClassifier(BaseDetector[ScanPayload, PredictionResult]):
    """
    Future: fine-tuned RoBERTa / DeBERTa-v3 for consent/subscription language.

    Until a checkpoint is loaded, returns status=not_loaded.
    """

    name = "text_classifier"
    version = "0.1.0"

    def __init__(self) -> None:
        self._model_path: str | None = None
        self._loaded = False

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(
            status="not_implemented",
            message="Text training pipeline not yet implemented. See docs/Training.md.",
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        if not self._loaded:
            pred = TextPrediction(
                status="not_loaded",
                message="Text model not loaded. Place checkpoint under models/text/ and call load_model().",
            )
            return PredictionResult(status="not_loaded", data=pred.model_dump(), message=pred.message)
        raise ModelNotLoadedError("Text inference runtime not implemented yet.")

    def classify(self, input_data: ScanPayload) -> TextPrediction:
        result = self.predict(input_data)
        return TextPrediction.model_validate(result.data)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="Text evaluation pending model integration.")

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Text model not found: {path}")
        self._model_path = str(p)
        self._loaded = False
        raise NotImplementedError("Text checkpoint found but HF inference is not wired yet.")

    def save_model(self, path: str) -> None:
        raise NotImplementedError("Text save_model is not implemented yet.")

    def is_ready(self) -> bool:
        return self._loaded
