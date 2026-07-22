"""Phase 4 semantic text classifier — Sentence Transformer + exemplar matching."""

from __future__ import annotations

import time
from typing import Any

from ai.common.base import BaseDetector
from ai.common.types import (
    Category,
    EvalMetrics,
    PredictionResult,
    ScanPayload,
    TextPrediction,
    TrainResult,
)
from ai.models.config import Phase4ModelConfig
from ai.text.embeddings.encoder import SentenceTransformerEngine
from ai.text.semantic.exemplars import PATTERN_TO_CATEGORY
from ai.text.semantic.similarity import collect_candidate_texts


class TextClassifier(BaseDetector[ScanPayload, PredictionResult]):
    """
    Pretrained semantic classifier for consent language.
    Does not replace rule explanations — returns structured NLP findings only.
    """

    name = "text_classifier"
    version = "0.4.0"

    def __init__(
        self,
        config: Phase4ModelConfig | None = None,
        embedding: SentenceTransformerEngine | None = None,
    ) -> None:
        self.config = config or Phase4ModelConfig.from_env()
        self._embedding = embedding or SentenceTransformerEngine(self.config)
        self._last_inference_ms = 0.0

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(
            status="not_applicable",
            message="Phase 4 uses pretrained sentence transformers; fine-tuning is optional later.",
        )

    def classify(self, input_data: ScanPayload) -> TextPrediction:
        t0 = time.perf_counter()
        if self.config.stub_mode and self.config.backend == "auto":
            return TextPrediction(
                status="not_loaded",
                message="Text model stub mode (AI_STUB_MODE=true). Set AI_STUB_MODE=false to enable Phase 4 NLP.",
            )

        candidates = collect_candidate_texts(input_data)
        if not candidates:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return TextPrediction(
                status="ready",
                category=Category.NO_DARK_PATTERN,
                confidence=0.0,
                evidence_spans=[],
                message="No consent-related text candidates collected.",
            )

        # Ground AI in collected DOM — require at least one consent-ish cue
        css = input_data.css_snapshot or {}
        has_dom_signal = bool(css.get("banner") or css.get("cmp") or css.get("buttons"))
        matches = self._embedding.match_exemplars(candidates)
        # Filter invented patterns: only keep matches tied to actual candidate text
        if not has_dom_signal:
            matches = [m for m in matches if m["pattern"] in ("confirmshaming",)]

        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        if not matches:
            return TextPrediction(
                status="ready",
                category=Category.NO_DARK_PATTERN,
                confidence=0.0,
                evidence_spans=[],
                message=f"NLP ready ({self._embedding._resolve_backend()}); no semantic matches above threshold.",
            )

        top = matches[0]
        pattern_cats = {
            "preselected_consent",
            "forced_action",
            "confirmshaming",
            "obstruction",
            "interface_interference",
        }
        dark = [m for m in matches if m["pattern"] in pattern_cats]
        if dark:
            top_dark = dark[0]
            cat_name = PATTERN_TO_CATEGORY.get(top_dark["pattern"], Category.COOKIE_CONSENT_MANIPULATION.value)
            category = Category.COOKIE_CONSENT_MANIPULATION
            for c in Category:
                if c.value == cat_name:
                    category = c
                    break
            conf = float(top_dark["confidence"])
        else:
            category = Category.NO_DARK_PATTERN
            conf = float(top["confidence"]) * 0.5  # UI element match, not a violation claim

        spans = [
            {
                "predicted_pattern": m["pattern"],
                "confidence": m["confidence"],
                "embedding_similarity": m["embedding_similarity"],
                "matched_example": m["matched_example"],
                "source_text": m["source_text"],
            }
            for m in matches[:8]
        ]
        return TextPrediction(
            status="ready",
            category=category,
            confidence=conf,
            evidence_spans=spans,
            message=(
                f"NLP backend={self._embedding._resolve_backend()} "
                f"top={top['pattern']} sim={top['confidence']:.2f} "
                f"inference_ms={self._last_inference_ms:.1f}"
            ),
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        pred = self.classify(input_data)
        return PredictionResult(status=pred.status, data=pred.model_dump(), message=pred.message)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="Use apps/api/tests/test_phase4_models.py.")

    def load_model(self, path: str) -> None:
        # Path override via env TEXT_MODEL_NAME / MODELS_ROOT — kept for BaseDetector API
        return None

    def save_model(self, path: str) -> None:
        return None

    def is_ready(self) -> bool:
        return self._embedding.is_ready()
