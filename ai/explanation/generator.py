"""Generate structured explainable reports from multimodal outputs."""

from __future__ import annotations

from typing import Any

from ai.common.base import BaseDetector
from ai.common.types import (
    Category,
    EvalMetrics,
    EvidenceItem,
    ExplainableReport,
    FusionOutput,
    PredictionResult,
    RuleResult,
    ScanPayload,
    TextPrediction,
    TrainResult,
    VisionFeatures,
)
from ai.evidence.enrichment import enrich_evidence_item
from ai.rules.evidence.builder import evidence_from_rules


class ExplanationGenerator(BaseDetector[FusionOutput, PredictionResult]):
    name = "explanation_generator"
    version = "0.2.0"

    def generate(
        self,
        fusion: FusionOutput,
        *,
        rules: RuleResult | None = None,
        vision: VisionFeatures | None = None,
        text: TextPrediction | None = None,
        payload: ScanPayload | None = None,
    ) -> ExplainableReport:
        evidence: list[EvidenceItem] = []
        notes: list[str] = []

        if rules:
            evidence.extend(evidence_from_rules(rules, payload))

        if vision:
            if vision.status != "ready":
                notes.append(
                    vision.message
                    or "Vision module did not contribute detections (model not loaded)."
                )
            elif vision.banner_detected:
                item = EvidenceItem(
                    id="vision:banner",
                    statement="Cookie/subscription banner detected by vision model.",
                    severity=0.4,
                    source="vision",
                )
                if payload is not None:
                    item = enrich_evidence_item(item, payload)
                evidence.append(item)

        if text:
            if text.status != "ready":
                notes.append(
                    text.message or "Text module did not contribute classifications (model not loaded)."
                )
            elif text.category and text.category != Category.NO_DARK_PATTERN:
                item = EvidenceItem(
                    id="text:category",
                    statement=f"Text model predicted: {text.category.value}",
                    severity=float(text.confidence or 0.5),
                    source="text",
                    metadata={"spans": text.evidence_spans},
                )
                if payload is not None:
                    item = enrich_evidence_item(item, payload)
                evidence.append(item)

        if fusion.message:
            notes.append(fusion.message)

        # Never return a bare "Dark Pattern Found" — always attach evidence or notes
        if fusion.category != Category.NO_DARK_PATTERN and not evidence:
            item = EvidenceItem(
                id="fusion:partial",
                statement="Category suggested by fusion with limited evidence; treat as provisional.",
                severity=0.3,
                source="fusion",
            )
            if payload is not None:
                item = enrich_evidence_item(item, payload)
            evidence.append(item)

        rule_traces = list(rules.traces) if rules else []

        return ExplainableReport(
            risk_score=fusion.risk_score,
            category=fusion.category,
            confidence=fusion.confidence,
            confidence_breakdown=fusion.confidence_breakdown,
            evidence=evidence,
            rule_traces=rule_traces,
            vision=vision,
            text=text,
            rules=rules,
            fusion=fusion,
            pipeline_notes=notes,
        )

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(status="not_applicable", message="Explanation module is deterministic.")

    def predict(self, input_data: FusionOutput) -> PredictionResult:
        report = self.generate(input_data)
        return PredictionResult(status="ready", data=report.model_dump())

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_applicable", message="N/A for explanation generator.")

    def load_model(self, path: str) -> None:
        return None

    def save_model(self, path: str) -> None:
        return None

    def is_ready(self) -> bool:
        return True
