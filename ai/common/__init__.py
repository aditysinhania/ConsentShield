"""Shared AI contracts, enums, and base detector interface."""

from ai.common.base import BaseDetector, ModelNotLoadedError, StubPredictionError
from ai.common.types import (
    Category,
    Confidence,
    EvidenceItem,
    ExplainableReport,
    FusionInput,
    FusionOutput,
    PredictionResult,
    RuleResult,
    ScanPayload,
    TextPrediction,
    TrainResult,
    EvalMetrics,
    VisionFeatures,
)

__all__ = [
    "BaseDetector",
    "ModelNotLoadedError",
    "StubPredictionError",
    "Category",
    "Confidence",
    "EvidenceItem",
    "ExplainableReport",
    "FusionInput",
    "FusionOutput",
    "PredictionResult",
    "RuleResult",
    "ScanPayload",
    "TextPrediction",
    "TrainResult",
    "EvalMetrics",
    "VisionFeatures",
]
