"""Phase 3.1 report enrichment — severity, clustering, accessibility."""

from __future__ import annotations

from ai.accessibility.analyzer import analyze_accessibility
from ai.clustering.patterns import cluster_evidence
from ai.common.types import ExplainableReport, FusionOutput, ScanPayload
from ai.severity.engine import compute_severity
from ai.report.ai_analysis import attach_ai_analysis


def enrich_report(
    report: ExplainableReport,
    *,
    payload: ScanPayload,
    fusion: FusionOutput,
) -> ExplainableReport:
    """Attach severity tier, pattern clusters, accessibility, and AI analysis."""
    accessibility = analyze_accessibility(payload)
    cmp_certainty = 0.0
    if fusion.confidence_breakdown is not None:
        cmp_certainty = fusion.confidence_breakdown.cmp

    rule_count = len(report.evidence)
    if report.rules and report.rules.hits:
        rule_count = len(report.rules.hits)

    severity = compute_severity(
        risk=report.risk_score,
        confidence=report.confidence,
        rule_count=rule_count,
        cmp_certainty=cmp_certainty,
        accessibility_issue_count=accessibility.issue_count,
        accessibility_score=accessibility.aggregate_score,
    )
    clusters = cluster_evidence(report.evidence)

    enriched = report.model_copy(
        update={
            "severity": severity.model_dump(),
            "pattern_clusters": [c.model_dump() for c in clusters],
            "accessibility": accessibility.model_dump(),
        }
    )
    return attach_ai_analysis(
        enriched,
        text=report.text,
        vision=report.vision,
        fusion=fusion,
    )
