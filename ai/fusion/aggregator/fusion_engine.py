"""Rule-dominant fusion aggregator until learned models are available."""

from __future__ import annotations

from typing import Any

from ai.common.base import BaseDetector
from ai.common.types import (
    Category,
    EvalMetrics,
    FusionInput,
    FusionOutput,
    PredictionResult,
    TrainResult,
)
from ai.fusion.classifier.model import FusionClassifier
from ai.fusion.feature_builder.builder import build_features


_CATEGORY_PRIORITY = [
    Category.MIXED_CONSENT_MANIPULATION,
    Category.COOKIE_CONSENT_MANIPULATION,
    Category.HIDDEN_SUBSCRIPTION,
    Category.HIDDEN_BILLING,
    Category.MISLEADING_FREE_TRIAL,
    Category.CONFIRMSHAMING,
]


def _map_category(names: list[str]) -> Category:
    if not names:
        return Category.NO_DARK_PATTERN
    normalized = set(names)
    if len(normalized) > 1:
        return Category.MIXED_CONSENT_MANIPULATION
    only = next(iter(normalized))
    for cat in Category:
        if cat.value == only:
            return cat
    return Category.UNKNOWN


class FusionEngine(BaseDetector[FusionInput, PredictionResult]):
    name = "fusion_engine"
    version = "0.1.0"

    def __init__(self) -> None:
        self._clf = FusionClassifier()

    def fuse(self, inputs: FusionInput) -> FusionOutput:
        features = build_features(inputs)
        sources: list[str] = []
        notes: list[str] = []

        if inputs.rules and inputs.rules.status == "ready":
            sources.append("rules")
        if inputs.vision:
            sources.append("vision")
            if inputs.vision.status != "ready":
                notes.append(f"Vision: {inputs.vision.status}")
        if inputs.text:
            sources.append("text")
            if inputs.text.status != "ready":
                notes.append(f"Text: {inputs.text.status}")

        # Prefer learned classifier when ready; otherwise rule-based aggregation
        if self._clf.is_ready():
            proba = self._clf.predict_proba(features)
            return FusionOutput(
                status="ready",
                category=Category.UNKNOWN,
                risk_score=0.0,
                confidence=0.0,
                feature_vector=features,
                sources_used=sources,
                message=str(proba.get("message")),
            )

        rules = inputs.rules
        if rules is None:
            return FusionOutput(
                status="partial",
                category=Category.UNKNOWN,
                risk_score=0.0,
                confidence=0.0,
                feature_vector=features,
                sources_used=sources,
                message="No rule results available; vision/text models not loaded.",
            )

        category = _map_category(rules.categories_triggered)
        # Confidence reflects rule coverage only until ML joins
        confidence = min(0.95, 0.35 + 0.1 * len(rules.hits)) if rules.hits else 0.7
        message = (
            "Fusion using rule engine only (vision/text stubs not loaded)."
            if notes
            else "Fusion using rule engine."
        )
        if notes:
            message = message + " " + " ".join(notes)

        return FusionOutput(
            status="ready",
            category=category,
            risk_score=rules.normalized_risk,
            confidence=round(confidence, 3),
            feature_vector=features,
            sources_used=sources,
            message=message,
        )

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(status="not_implemented", message="Train fusion head via ai/training.")

    def predict(self, input_data: FusionInput) -> PredictionResult:
        out = self.fuse(input_data)
        return PredictionResult(status=out.status, data=out.model_dump(), message=out.message)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="Fusion evaluation pending.")

    def load_model(self, path: str) -> None:
        raise NotImplementedError("Learned fusion load_model not implemented yet.")

    def save_model(self, path: str) -> None:
        raise NotImplementedError("Learned fusion save_model not implemented yet.")

    def is_ready(self) -> bool:
        return True  # rule-based path always available
