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


class RuleExplanation(BaseModel):
    """Deterministic structured explanation for a triggered rule."""

    explanation: str
    user_impact: str
    gdpr_relevance: str
    recommendation: str


class BoundingRect(BaseModel):
    x: float
    y: float
    width: float
    height: float
    top: float
    left: float
    right: float
    bottom: float


class ConfidenceBreakdown(BaseModel):
    """Component confidence scores; final mirrors FusionOutput.confidence."""

    text: float = Field(ge=0.0, le=1.0, default=0.0)
    visual: float = Field(ge=0.0, le=1.0, default=0.0)
    layout: float = Field(ge=0.0, le=1.0, default=0.0)
    cmp: float = Field(ge=0.0, le=1.0, default=0.0)
    agreement: float = Field(ge=0.0, le=1.0, default=0.0)
    final: float = Field(ge=0.0, le=1.0, default=0.0)
    # Phase 4 contribution channels (additive; rules remain primary)
    rules: float = Field(ge=0.0, le=1.0, default=0.0)
    nlp: float = Field(ge=0.0, le=1.0, default=0.0)
    vision_model: float = Field(ge=0.0, le=1.0, default=0.0)
    fusion: float = Field(ge=0.0, le=1.0, default=0.0)


class RuleTrace(BaseModel):
    """Per-rule execution traceability record."""

    rule_id: str
    features_used: list[str] = Field(default_factory=list)
    visual_score: float = Field(ge=0.0, le=1.0, default=0.0)
    text_score: float = Field(ge=0.0, le=1.0, default=0.0)
    layout_score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    risk_contribution: float = Field(ge=0.0, le=100.0, default=0.0)


class EvidenceItem(BaseModel):
    id: str
    statement: str
    severity: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str  # rules | vision | text | fusion | dom
    rule_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Structured explanation (deterministic)
    explanation: str | None = None
    user_impact: str | None = None
    gdpr_relevance: str | None = None
    recommendation: str | None = None
    # Evidence enrichment — only set when real values are available
    xpath: str | None = None
    css_selector: str | None = None
    dom_path: str | None = None
    bounding_rect: BoundingRect | None = None
    viewport: dict[str, int] | None = None
    scroll_position: dict[str, int] | None = None
    html_snippet: str | None = None
    computed_styles: dict[str, Any] | None = None
    timestamp: str | None = None
    url: str | None = None
    page_title: str | None = None


class ScanPayload(BaseModel):
    """Payload collected by the Chrome extension."""

    url: str
    title: str | None = None
    html: str | None = None
    css_snapshot: dict[str, Any] | None = None
    visible_text: str | None = None
    screenshot_base64: str | None = None
    viewport: dict[str, int] | None = None
    scroll_position: dict[str, int] | None = None
    collected_at: str | None = None
    collection_duration_ms: float | None = None


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
    features_used: list[str] = Field(default_factory=list)
    visual_score: float = Field(ge=0.0, le=1.0, default=0.0)
    text_score: float = Field(ge=0.0, le=1.0, default=0.0)
    layout_score: float = Field(ge=0.0, le=1.0, default=0.0)


class RuleResult(BaseModel):
    status: str = "ready"
    hits: list[RuleHit] = Field(default_factory=list)
    total_score: float = 0.0
    normalized_risk: float = Field(ge=0.0, le=100.0, default=0.0)
    categories_triggered: list[str] = Field(default_factory=list)
    traces: list[RuleTrace] = Field(default_factory=list)


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
    confidence_breakdown: ConfidenceBreakdown | None = None
    feature_vector: list[float] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    message: str | None = None


class ExplainableReport(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    rule_traces: list[RuleTrace] = Field(default_factory=list)
    severity: dict[str, Any] | None = None
    pattern_clusters: list[dict[str, Any]] = Field(default_factory=list)
    accessibility: dict[str, Any] | None = None
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    performance: dict[str, Any] | None = None
    annotated_screenshot_path: str | None = None
    narrator: dict[str, Any] | None = None
    vision: VisionFeatures | None = None
    text: TextPrediction | None = None
    rules: RuleResult | None = None
    fusion: FusionOutput | None = None
    pipeline_notes: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
    ai_analysis: dict[str, Any] | None = None


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
