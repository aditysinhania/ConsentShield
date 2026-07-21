"""Deterministic severity tier engine (LOW / MEDIUM / HIGH / CRITICAL)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SeverityFactors(BaseModel):
    risk: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rule_count: int = Field(ge=0)
    cmp_certainty: float = Field(ge=0.0, le=1.0)
    accessibility_score: float = Field(ge=0.0, le=1.0, description="0=none, 1=critical a11y burden")
    accessibility_issue_count: int = Field(ge=0, default=0)


class SeverityAssessment(BaseModel):
    level: SeverityLevel
    composite_score: float = Field(ge=0.0, le=100.0)
    justification: str
    factors: SeverityFactors


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_severity(
    *,
    risk: float,
    confidence: float,
    rule_count: int,
    cmp_certainty: float,
    accessibility_issue_count: int = 0,
    accessibility_score: float = 0.0,
) -> SeverityAssessment:
    """
    Derive tier from risk, confidence, rule count, CMP certainty, and accessibility.
    Rules remain source of truth; severity summarizes audit urgency.
    """
    risk = _clamp(risk, 0.0, 100.0)
    confidence = _clamp(confidence, 0.0, 1.0)
    cmp_certainty = _clamp(cmp_certainty, 0.0, 1.0)
    accessibility_score = _clamp(accessibility_score, 0.0, 1.0)
    rule_count = max(0, rule_count)

    rule_factor = min(rule_count / 4.0, 1.0) * 100.0
    a11y_factor = max(accessibility_score * 100.0, min(accessibility_issue_count * 15.0, 60.0))

    composite = (
        0.45 * risk
        + 0.15 * (confidence * 100.0)
        + 0.15 * rule_factor
        + 0.10 * (cmp_certainty * 100.0)
        + 0.15 * a11y_factor
    )
    composite = round(_clamp(composite, 0.0, 100.0), 2)

    factors = SeverityFactors(
        risk=risk,
        confidence=confidence,
        rule_count=rule_count,
        cmp_certainty=cmp_certainty,
        accessibility_score=accessibility_score,
        accessibility_issue_count=accessibility_issue_count,
    )

    if composite >= 72.0 or (risk >= 65.0 and rule_count >= 3 and accessibility_issue_count >= 2):
        level = SeverityLevel.CRITICAL
    elif composite >= 48.0 or (risk >= 38.0 and rule_count >= 2):
        level = SeverityLevel.HIGH
    elif composite >= 22.0 or risk >= 12.0 or rule_count >= 1:
        level = SeverityLevel.MEDIUM
    else:
        level = SeverityLevel.LOW

    parts = [
        f"composite score {composite}/100",
        f"risk {risk:.0f}",
        f"confidence {confidence:.0%}",
        f"{rule_count} rule(s) triggered",
        f"CMP certainty {cmp_certainty:.0%}",
    ]
    if accessibility_issue_count:
        parts.append(f"{accessibility_issue_count} accessibility issue(s)")
    justification = f"Severity {level.value}: " + "; ".join(parts) + "."

    return SeverityAssessment(
        level=level,
        composite_score=composite,
        justification=justification,
        factors=factors,
    )
