"""Phase 3.3 — multi-format report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.datasets.fixtures.validation_sites import (
    adobe_payload,
    guardian_payload,
    mozilla_payload,
    reuters_payload,
    spotify_payload,
)
from ai.inference.pipeline import InferencePipeline
from ai.report.document import build_report_document
from ai.report.exporters import export_html, export_json, export_markdown, export_pdf, write_exports


@pytest.fixture(scope="module")
def pipeline() -> InferencePipeline:
    return InferencePipeline()


def _document_for(payload, report):
    return build_report_document(
        scan_id="test-scan",
        url=payload.url,
        title=payload.title,
        risk_score=report.risk_score,
        category=report.category.value,
        confidence=report.confidence,
        report=report.model_dump(),
    )


def _assert_sections(doc) -> None:
    assert doc.executive_summary
    assert isinstance(doc.findings, list)
    assert isinstance(doc.evidence, list)
    assert doc.screenshots is not None
    assert doc.accessibility is not None or doc.accessibility is None
    assert isinstance(doc.timeline, list)
    assert doc.performance is not None or doc.performance is None
    assert isinstance(doc.recommendations, list)
    assert doc.appendix


@pytest.mark.parametrize(
    "payload_fn,site",
    [
        (guardian_payload, "Guardian"),
        (adobe_payload, "Adobe"),
        (spotify_payload, "Spotify"),
        (mozilla_payload, "Mozilla"),
        (reuters_payload, "Reuters"),
    ],
)
def test_generate_reports_all_formats(pipeline, payload_fn, site, tmp_path: Path):
    payload = payload_fn()
    report = pipeline.run(payload)
    doc = _document_for(payload, report)
    _assert_sections(doc)

    j = export_json(doc)
    h = export_html(doc)
    m = export_markdown(doc)
    p = export_pdf(doc)

    assert site in doc.executive_summary or payload.url in doc.executive_summary
    assert "Executive Summary" in h
    assert "## Executive Summary" in m
    assert len(p) > 100
    assert "ConsentShield" in h

    paths = write_exports(doc, tmp_path / site)
    assert paths["json"] and Path(paths["json"]).is_file()
    assert paths["html"] and Path(paths["html"]).is_file()
    assert paths["markdown"] and Path(paths["markdown"]).is_file()
    assert paths["pdf"] and Path(paths["pdf"]).stat().st_size > 0


def test_markdown_includes_required_sections(pipeline):
    report = pipeline.run(guardian_payload())
    doc = _document_for(guardian_payload(), report)
    md = export_markdown(doc)
    for heading in (
        "## Executive Summary",
        "## Findings",
        "## Evidence",
        "## Screenshots",
        "## Accessibility",
        "## Timeline",
        "## Performance",
        "## Recommendations",
        "## Appendix",
    ):
        assert heading in md
