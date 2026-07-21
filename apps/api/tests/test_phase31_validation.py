"""Phase 3.1 validation — severity, clustering, accessibility for reference sites."""

from __future__ import annotations

import pytest

from ai.accessibility.analyzer import analyze_accessibility
from ai.clustering.patterns import PatternClusterName, cluster_evidence
from ai.inference.pipeline import InferencePipeline
from ai.rules.evidence.builder import evidence_from_rules
from ai.severity.engine import SeverityLevel
from ai.datasets.fixtures.validation_sites import (
    adobe_payload,
    cookielawinfo_payload,
    guardian_payload,
    mozilla_payload,
    reuters_payload,
    spotify_payload,
)


@pytest.fixture(scope="module")
def pipeline() -> InferencePipeline:
    return InferencePipeline()


def _assert_phase31_fields(report) -> None:
    assert report.severity is not None
    assert report.severity["level"] in {s.value for s in SeverityLevel}
    assert report.severity["justification"]
    assert report.severity["factors"]["risk"] == pytest.approx(report.risk_score, abs=0.01)
    assert isinstance(report.pattern_clusters, list)
    assert report.accessibility is not None
    assert report.accessibility["status"] == "ready"
    assert "issues" in report.accessibility
    # Accessibility stored separately — not mixed into evidence sources
    assert all(e.source != "accessibility" for e in report.evidence)


def test_guardian_validation(pipeline: InferencePipeline):
    report = pipeline.run(guardian_payload())
    _assert_phase31_fields(report)
    assert 30 <= report.risk_score <= 55
    assert 0.55 <= report.confidence <= 0.85
    assert report.severity["level"] in {SeverityLevel.MEDIUM.value, SeverityLevel.HIGH.value}
    clusters = {c["cluster"] for c in report.pattern_clusters}
    assert PatternClusterName.VISUAL_MANIPULATION.value in clusters or PatternClusterName.HIDDEN_CHOICE.value in clusters


def test_adobe_validation(pipeline: InferencePipeline):
    report = pipeline.run(adobe_payload())
    _assert_phase31_fields(report)
    assert 28 <= report.risk_score <= 55
    clusters = {c["cluster"] for c in report.pattern_clusters}
    assert (
        PatternClusterName.PRIVACY_FRICTION.value in clusters
        or PatternClusterName.VISUAL_MANIPULATION.value in clusters
        or PatternClusterName.HIDDEN_CHOICE.value in clusters
    )


def test_spotify_validation(pipeline: InferencePipeline):
    report = pipeline.run(spotify_payload())
    _assert_phase31_fields(report)
    assert report.risk_score <= 20
    assert report.severity["level"] == SeverityLevel.LOW.value
    assert len(report.pattern_clusters) <= 1


def test_mozilla_validation(pipeline: InferencePipeline):
    report = pipeline.run(mozilla_payload())
    _assert_phase31_fields(report)
    assert 12 <= report.risk_score <= 40
    assert report.severity["level"] in {SeverityLevel.LOW.value, SeverityLevel.MEDIUM.value}


def test_reuters_validation(pipeline: InferencePipeline):
    report = pipeline.run(reuters_payload())
    _assert_phase31_fields(report)
    assert report.risk_score >= 20
    clusters = {c["cluster"] for c in report.pattern_clusters}
    assert PatternClusterName.HIDDEN_CHOICE.value in clusters


def test_cookielawinfo_validation(pipeline: InferencePipeline):
    report = pipeline.run(cookielawinfo_payload())
    _assert_phase31_fields(report)
    assert 20 <= report.risk_score <= 50
    clusters = {c["cluster"] for c in report.pattern_clusters}
    assert (
        PatternClusterName.FORCED_ACTION.value in clusters
        or PatternClusterName.OBSTRUCTION.value in clusters
    )
    assert report.accessibility["issue_count"] >= 0


def test_clustering_groups_rule_hits():
    from ai.rules import RuleEngine

    result = RuleEngine().evaluate_payload(guardian_payload())
    items = evidence_from_rules(result, guardian_payload())
    clusters = cluster_evidence(items)
    assert clusters
    assert all(c.count == len(c.finding_ids) for c in clusters)


def test_accessibility_detects_overlay():
    report = analyze_accessibility(guardian_payload())
    types = {i.type.value for i in report.issues}
    assert "overlay_blocking" in types or "focus_trap" in types
