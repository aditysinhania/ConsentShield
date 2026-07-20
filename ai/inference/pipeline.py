"""End-to-end inference orchestration."""

from __future__ import annotations

from ai.common.types import ExplainableReport, FusionInput, ScanPayload
from ai.explanation.generator import ExplanationGenerator
from ai.fusion.aggregator.fusion_engine import FusionEngine
from ai.rules import RuleEngine
from ai.text.classifier.text_classifier import TextClassifier
from ai.vision.detectors.vision_detector import VisionDetector


class InferencePipeline:
    """
    Chrome Extension payload → Rules + Vision stub + Text stub → Fusion → Explanation.
    """

    def __init__(self) -> None:
        self.rules = RuleEngine()
        self.vision = VisionDetector()
        self.text = TextClassifier()
        self.fusion = FusionEngine()
        self.explainer = ExplanationGenerator()

    def run(self, payload: ScanPayload) -> ExplainableReport:
        rule_result = self.rules.evaluate_payload(payload)
        vision_features = self.vision.extract_features(payload)
        text_pred = self.text.classify(payload)

        fusion_out = self.fusion.fuse(
            FusionInput(
                vision=vision_features,
                text=text_pred,
                rules=rule_result,
                dom_features=payload.css_snapshot or {},
            )
        )
        return self.explainer.generate(
            fusion_out,
            rules=rule_result,
            vision=vision_features,
            text=text_pred,
        )
