"""Phase 3.5 regression tests and performance benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.datasets.fixtures.validation_sites import (
    adobe_payload,
    bbc_payload,
    guardian_payload,
    mozilla_payload,
    reuters_payload,
    spotify_payload,
)
from ai.evidence.quality import apply_evidence_quality, score_evidence_item
from ai.inference.pipeline import InferencePipeline
from ai.performance.instrumentation import benchmark_pipeline
from ai.registry.model_registry import ModelRegistry
from ai.registry.interfaces import EmbeddingEngine, ReasoningEngine, TextClassifier, VisionClassifier
from ai.report.document import build_report_document
from ai.report.exporters import export_html, export_pdf
from ai.rules.evidence.builder import evidence_from_rules
from ai.rules import RuleEngine
from ai.screenshots.annotator import annotate_screenshot


@pytest.fixture(scope="module")
def pipeline() -> InferencePipeline:
    return InferencePipeline()


SITES = [
    ("Guardian", guardian_payload),
    ("Adobe", adobe_payload),
    ("Spotify", spotify_payload),
    ("Mozilla", mozilla_payload),
    ("Reuters", reuters_payload),
    ("BBC", bbc_payload),
]


@pytest.mark.parametrize("name,payload_fn", SITES)
def test_regression_risk_scores_stable(name, payload_fn, pipeline):
    """Phase 2B risk scores must remain in expected bands."""
    report = pipeline.run(payload_fn())
    bands = {
        "Guardian": (30, 55),
        "Adobe": (28, 55),
        "Spotify": (0, 25),
        "Mozilla": (12, 40),
        "Reuters": (20, 55),
        "BBC": (20, 55),
    }
    lo, hi = bands[name]
    assert lo <= report.risk_score <= hi, f"{name} risk {report.risk_score} outside [{lo},{hi}]"
    assert report.severity is not None
    assert report.pattern_clusters is not None
    assert report.accessibility is not None
    assert report.performance.get("pipeline_version") == "0.5.0"


@pytest.mark.parametrize("name,payload_fn", SITES)
def test_regression_evidence_quality(name, payload_fn):
    payload = payload_fn()
    result = RuleEngine().evaluate_payload(payload)
    items = evidence_from_rules(result, payload)
    if not items:
        pytest.skip(f"{name} has no rule hits — evidence quality N/A")
    for item in items:
        assert item.metadata.get("quality_score") is not None
        assert item.metadata.get("quality_tier") in ("high", "medium", "low")
        assert score_evidence_item(item) >= 0.0


def test_model_registry_di_interfaces():
    reg = ModelRegistry.default()
    assert isinstance(reg.text, TextClassifier)
    assert isinstance(reg.vision, VisionClassifier)
    assert isinstance(reg.embedding, EmbeddingEngine)
    assert isinstance(reg.reasoning, ReasoningEngine)
    desc = reg.describe()
    assert len(desc) >= 4
    assert all("interface" in d for d in desc)


def test_professional_html_report(pipeline):
    report = pipeline.run(guardian_payload())
    doc = build_report_document(
        scan_id="regression",
        url=guardian_payload().url,
        title="Guardian",
        risk_score=report.risk_score,
        category=report.category.value,
        confidence=report.confidence,
        report=report.model_dump(),
    )
    html = export_html(doc)
    assert "ConsentShield · Consent Compliance Audit" in html
    assert "section-num" in html
    assert "risk-dashboard" in html
    assert "Evidence quality" in html or "quality" in html.lower()
    assert len(export_pdf(doc)) > 200


def test_annotated_screenshot_regression(pipeline, tmp_path: Path):
    report = pipeline.run(guardian_payload())
    out = tmp_path / "annotated.png"
    annotate_screenshot(source_path=None, output_path=out, payload=guardian_payload(), report=report)
    assert out.exists()
    assert out.stat().st_size > 500


def test_performance_benchmark():
    result = benchmark_pipeline(guardian_payload(), runs=2)
    assert result["runs"] == 2
    assert result["total_ms"]["mean"] >= 0
    assert "rule_engine" in result["stages"]
    assert result["pipeline_version"] == "0.5.0"
