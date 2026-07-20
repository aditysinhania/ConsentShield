"""Scoring utilities for rule hits."""

from __future__ import annotations

from ai.common.types import RuleHit, RuleResult


def score_hits(hits: list[RuleHit]) -> RuleResult:
    if not hits:
        return RuleResult(
            status="ready",
            hits=[],
            total_score=0.0,
            normalized_risk=0.0,
            categories_triggered=[],
        )

    total = sum(h.score * max(h.severity, 0.1) for h in hits)
    # Soft saturation toward 100
    normalized = min(100.0, round(100.0 * (1.0 - pow(2.718281828, -total / 2.5)), 2))
    categories = sorted({h.category for h in hits})
    return RuleResult(
        status="ready",
        hits=hits,
        total_score=round(total, 4),
        normalized_risk=normalized,
        categories_triggered=categories,
    )
