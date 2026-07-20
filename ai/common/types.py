"""Shared typed contracts for the multimodal pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Category(str, Enum):
    NO_DARK_PATTERN = "No Dark Pattern"
    COOKIE_CONSENT_MANIPULATION = "Cookie Consent Manipulation"
    HIDDEN_SUBSCRIPTION = "Hidden Subscription"
    HIDDEN_BILLING = "Hidden Billing"
    CONFIRMSHAMING = "Confirmshaming"
    MISLEADING_FREE_TRIAL = "Misleading Free Trial"
    MIXED_CONSENT_MANIPULATION = "Mixed Consent Manipulation"
    UNKNOWN = "Unknown"


class Confidence(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    source: str = "unknown"


class EvidenceItem(BaseModel):
    id: str
    statement: str
    severity: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str  # rules | vision | text | fusion | dom
    rule_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScanPayload(BaseModel):
    """Payload collected by the Chrome extension."""

    url: str
    title: str | None = None
    html: str | None = None
    css_snapshot: dict[str, Any] | None = None
    visible_text: str | None = None
    screenshot_base64: str | None = None
    viewport: dict[str, int] | None = None
    collected_at: str | None = None


class VisionFeatures(BaseModel):
    status: str  # ready | stub | not_loaded | error
    banner_detected: bool | None = None
    accept_button: dict[str, Any] | None = None
    reject_button: dict[str, Any] | None = None
    checkboxes: list[dict[str, Any]] = Field(default_factory=list)
    toggles: list[dict[str, Any]] = Field(default_factory=list)
    layout: dict[str, Any] = Field(default_factory=dict)
    feature_vector: list[float] = Field(default_factory=list)
    message: str | None = None


class TextPrediction(BaseModel):
    status: str
    category: Category | None = None
    confidence: float | None = None
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None


class RuleHit(BaseModel):
    rule_id: str
    name: str
    category: str
    severity: float
    score: float
    evidence: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuleResult(BaseModel):
    status: str = "ready"
    hits: list[RuleHit] = Field(default_factory=list)
    total_score: float = 0.0
    normalized_risk: float = Field(ge=0.0, le=100.0, default=0.0)
    categories_triggered: list[str] = Field(default_factory=list)


class FusionInput(BaseModel):
    vision: VisionFeatures | None = None
    text: TextPrediction | None = None
    rules: RuleResult | None = None
    dom_features: dict[str, Any] = Field(default_factory=dict)


class FusionOutput(BaseModel):
    status: str
    category: Category
    risk_score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    feature_vector: list[float] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    message: str | None = None


class ExplainableReport(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    vision: VisionFeatures | None = None
    text: TextPrediction | None = None
    rules: RuleResult | None = None
    fusion: FusionOutput | None = None
    pipeline_notes: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class TrainResult(BaseModel):
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_path: str | None = None
    message: str | None = None


class EvalMetrics(BaseModel):
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str | None = None
