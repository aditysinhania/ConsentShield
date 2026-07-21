"""Phase 3.2 — timeline, performance metrics, annotated screenshots."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from ai.common.types import ScanPayload
from ai.datasets.fixtures.validation_sites import guardian_payload
from ai.inference.pipeline import InferencePipeline
from ai.performance.metrics import MetricStage, PerformanceCollector
from ai.screenshots.annotator import annotate_screenshot
from ai.timeline.recorder import TimelineEventName, TimelineRecorder


def _minimal_png_b64(width: int = 400, height: int = 300) -> str:
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_timeline_json_events():
    payload = guardian_payload()
    timeline = TimelineRecorder()
    timeline.mark(TimelineEventName.PAGE_LOADED, at=payload.collected_at or "2026-07-21T10:00:00Z")
    report = InferencePipeline().run(payload, timeline=timeline)
    events = report.timeline
    assert events
    names = {e["event"] for e in events}
    assert "page_loaded" in names
    assert "rule_engine" in names
    assert "fusion" in names
    assert "explanation" in names
    for e in events:
        assert "timestamp" in e
        assert e["timestamp"]


def test_performance_metrics():
    payload = guardian_payload()
    metrics = PerformanceCollector()
    metrics.set(MetricStage.COLLECTION, 42.5)
    report = InferencePipeline().run(payload, metrics=metrics)
    perf = report.performance
    assert perf is not None
    assert perf["collection_ms"] == 42.5
    assert perf["rule_engine_ms"] >= 0
    assert perf["fusion_ms"] >= 0
    assert perf["report_ms"] >= 0
    assert perf["llm_ms"] == 0.0
    assert "stages" in perf


def test_annotated_screenshot_generated(tmp_path: Path):
    payload = guardian_payload()
    payload = ScanPayload(
        **{
            **payload.model_dump(),
            "screenshot_base64": _minimal_png_b64(640, 480),
        }
    )
    raw = tmp_path / "raw.png"
    raw.write_bytes(base64.b64decode(payload.screenshot_base64.split(",", 1)[1]))

    report = InferencePipeline().run(payload)
    out = tmp_path / "annotated.png"
    result = annotate_screenshot(
        source_path=raw,
        output_path=out,
        payload=payload,
        report=report,
    )
    assert result.exists()
    assert result.stat().st_size > 0
    with Image.open(result) as img:
        assert img.format == "PNG"
        assert img.height > 480  # footer added


def test_pipeline_includes_timeline_and_metrics():
    report = InferencePipeline().run(guardian_payload())
    assert report.timeline
    assert report.performance
    assert report.performance["rule_engine_ms"] >= 0


def test_banner_and_cmp_timeline_events():
    payload = guardian_payload()
    timeline = TimelineRecorder()
    timeline.mark(TimelineEventName.PAGE_LOADED, duration_ms=0.0)
    report = InferencePipeline().run(payload, timeline=timeline)
    names = [e["event"] for e in report.timeline]
    assert "banner_found" in names
    assert "cmp_found" in names
