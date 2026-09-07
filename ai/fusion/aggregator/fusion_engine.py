"""Rule-dominant fusion aggregator with Phase 4 NLP/vision assistance."""

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
from ai.fusion.confidence import compute_confidence_breakdown
from ai.fusion.feature_builder.builder import build_features


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


def _dom_supports_ai(dom: dict[str, Any]) -> bool:
    if not dom:
        return False
    if isinstance(dom.get("banner"), dict) and (
        dom["banner"].get("width") or dom["banner"].get("height") or dom["banner"].get("xpath")
    ):
        return True
    if isinstance(dom.get("cmp"), dict) and dom["cmp"].get("detected"):
        return True
    buttons = dom.get("buttons") or []
    return any(isinstance(b, dict) and (b.get("text") or b.get("ariaLabel")) for b in buttons)


class FusionEngine(BaseDetector[FusionInput, PredictionResult]):
    name = "fusion_engine"
    version = "0.5.0"

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
            elif inputs.vision.message:
                notes.append(inputs.vision.message)
        if inputs.text:
            sources.append("text")
            if inputs.text.status != "ready":
                notes.append(f"Text: {inputs.text.status}")
            elif inputs.text.message:
                notes.append(inputs.text.message)

        if self._clf.is_ready():
            proba = self._clf.predict_proba(features)
            confidence = 0.0
            breakdown = compute_confidence_breakdown(inputs, final_confidence=confidence)
            return FusionOutput(
                status="ready",
                category=Category.UNKNOWN,
                risk_score=0.0,
                confidence=confidence,
                confidence_breakdown=breakdown,
                feature_vector=features,
                sources_used=sources,
                message=str(proba.get("message")),
            )

        rules = inputs.rules
        if rules is None:
            confidence = 0.0
            breakdown = compute_confidence_breakdown(inputs, final_confidence=confidence)
            return FusionOutput(
                status="partial",
                category=Category.UNKNOWN,
                risk_score=0.0,
                confidence=confidence,
                confidence_breakdown=breakdown,
                feature_vector=features,
                sources_used=sources,
                message="No rule results available; vision/text models not loaded.",
            )

        # Rules are the source of truth for risk score
        risk = float(rules.normalized_risk)
        category = _map_category(rules.categories_triggered)

        confidence = min(0.95, 0.35 + 0.1 * len(rules.hits)) if rules.hits else 0.7

        # MiniLM confidence may corroborate rule hits / boost confidence only.
        # Never invent high risk without supporting evidence (rules or grounded DOM).
        nlp_ready = inputs.text and inputs.text.status == "ready"
        vision_ready = inputs.vision and inputs.vision.status == "ready"
        grounded = _dom_supports_ai(inputs.dom_features or {})
        nlp_conf = float(inputs.text.confidence or 0.0) if nlp_ready else 0.0
        nlp_dark = bool(
            nlp_ready
            and inputs.text.category
            and inputs.text.category != Category.NO_DARK_PATTERN
            and nlp_conf >= 0.55
        )

        if grounded and rules.hits:
            if nlp_ready and nlp_conf:
                confidence = min(0.95, confidence + 0.10 * nlp_conf)
            if vision_ready and inputs.vision.banner_detected:
                confidence = min(0.95, confidence + 0.05)

        # Soft corroboration only when rules already fired and AI agrees
        if rules.hits and grounded:
            ai_boost = 0.0
            if nlp_dark:
                if inputs.text.category.value in rules.categories_triggered or any(
                    "cookie" in c.lower() or "consent" in c.lower() for c in rules.categories_triggered
                ):
                    ai_boost += 2.5 * nlp_conf
            if vision_ready and inputs.vision.banner_detected:
                ai_boost += 1.5
            risk = min(100.0, risk + ai_boost)

        # Text-only dark patterns: DOM-grounded MiniLM signal, no rule hits.
        # Cap risk low — never invent high risk without rule corroboration.
        elif not rules.hits and grounded and nlp_dark and nlp_conf >= 0.70:
            category = inputs.text.category or Category.UNKNOWN
            risk = min(22.0, 8.0 + 12.0 * nlp_conf)
            confidence = min(0.70, 0.40 + 0.30 * nlp_conf)
            notes.append(
                "Text-only Fine-tuned MiniLM signal (DOM-grounded; low risk until rules corroborate)."
            )

        confidence = round(confidence, 3)
        breakdown = compute_confidence_breakdown(inputs, final_confidence=confidence)

        ai_bits = []
        if nlp_ready:
            backend = getattr(inputs.text, "backend", None) or "nlp"
            ai_bits.append("minilm" if backend == "finetuned_minilm" else "nlp")
        if vision_ready:
            ai_bits.append("vision")
        if ai_bits and rules.hits:
            message = f"Fusion rule-dominant with AI assist ({'+'.join(ai_bits)})."
        elif ai_bits and not rules.hits and nlp_dark:
            message = (
                f"Fusion text-only MiniLM assist ({'+'.join(ai_bits)}); "
                "rules remain source of truth for high risk."
            )
        elif ai_bits:
            message = f"Fusion using rules; AI channels ready ({'+'.join(ai_bits)}) without overriding risk."
        elif notes:
            message = "Fusion using rule engine only (vision/text stubs not loaded)."
        else:
            message = "Fusion using rule engine."
        if notes:
            message = message + " " + " ".join(notes[:3])

        return FusionOutput(
            status="ready",
            category=category,
            risk_score=round(risk, 2),
            confidence=confidence,
            confidence_breakdown=breakdown,
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
        return True
