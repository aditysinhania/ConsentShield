"""Scan processing service — orchestrates AI pipeline and persistence."""

from __future__ import annotations

import base64
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

# Ensure monorepo root is importable for the `ai` package
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.common.types import ScanPayload
from ai.inference.pipeline import InferencePipeline
from ai.narrator.narrator import LLMNarrator
from ai.performance.instrumentation import PerformanceInstrumentation
from ai.performance.metrics import MetricStage, PerformanceCollector
from ai.report.generator import generate_all_formats, generate_report_document
from ai.screenshots.annotator import annotate_screenshot
from ai.timeline.recorder import TimelineEventName, TimelineRecorder

from app.core.config import settings
from app.models.detection import DetectedPattern, ModelPrediction, RuleEngineResult
from app.models.scan import ScanResult, Screenshot, WebsiteScan

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


def _save_screenshot(scan_id: uuid.UUID, b64: str | None) -> tuple[str | None, float]:
    if not b64:
        return None, 0.0
    t0 = time.perf_counter()
    raw = b64.split(",", 1)[-1] if "," in b64 else b64
    data = base64.b64decode(raw)
    out_dir = Path(settings.SCREENSHOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scan_id}.png"
    path.write_bytes(data)
    return str(path), (time.perf_counter() - t0) * 1000.0


async def create_and_process_scan(
    db: AsyncSession,
    *,
    payload: ScanPayload,
    user_id: uuid.UUID | None,
) -> WebsiteScan:
    perf = PerformanceInstrumentation()
    perf.start_total()
    timeline = TimelineRecorder()

    if payload.collection_duration_ms is not None:
        perf.set(MetricStage.COLLECTION, float(payload.collection_duration_ms))
    else:
        perf.set(MetricStage.COLLECTION, 0.0)

    timeline.mark(
        TimelineEventName.PAGE_LOADED,
        at=payload.collected_at,
        duration_ms=0.0,
    )

    scan = WebsiteScan(
        user_id=user_id,
        url=payload.url,
        title=payload.title,
        status="processing",
        visible_text=payload.visible_text,
        html_excerpt=(payload.html or "")[:50_000] or None,
        css_snapshot=payload.css_snapshot,
    )
    db.add(scan)
    await db.flush()

    shot_path, screenshot_ms = _save_screenshot(scan.id, payload.screenshot_base64)
    perf.set(MetricStage.SCREENSHOT, screenshot_ms)
    if shot_path:
        db.add(Screenshot(scan_id=scan.id, storage_path=shot_path))

    report = get_pipeline().run(payload, timeline=timeline, metrics=perf)

    annotated_path: str | None = None
    t_ann = time.perf_counter()
    try:
        out_ann = Path(settings.SCREENSHOTS_DIR) / f"{scan.id}_annotated.png"
        annotate_screenshot(
            source_path=shot_path,
            output_path=out_ann,
            payload=payload,
            report=report,
        )
        annotated_path = str(out_ann)
    except Exception:
        annotated_path = None
    perf.set(MetricStage.ANNOTATION, (time.perf_counter() - t_ann) * 1000.0)

    total_ms = perf.finish_total()
    timeline.mark(TimelineEventName.REPORT, metadata={"total_ms": total_ms})

    perf_data = perf.finalize()
    perf_data["llm_ms"] = perf_data.get("llm_ms", 0.0)

    report = report.model_copy(
        update={
            "timeline": timeline.to_list(),
            "performance": perf_data,
            "annotated_screenshot_path": annotated_path,
        }
    )

    report_payload = report.model_dump()

    narrator = LLMNarrator.from_env(
        provider=settings.AI_PROVIDER,
        gemini_api_key=settings.GEMINI_API_KEY,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    narrator_result = narrator.narrate(report_payload, url=scan.url)
    report_payload["narrator"] = narrator_result.model_dump()
    perf_data = dict(report_payload.get("performance") or {})
    perf_data["llm_ms"] = narrator_result.duration_ms
    perf_data.setdefault("stages", {})["llm"] = narrator_result.duration_ms
    report_payload["performance"] = perf_data

    try:
        doc = generate_report_document(
            scan_id=str(scan.id),
            url=scan.url,
            title=scan.title,
            risk_score=report.risk_score,
            category=report.category.value,
            confidence=report.confidence,
            report=report_payload,
            screenshot_path=shot_path,
            annotated_screenshot_path=annotated_path,
            narrator=report_payload["narrator"],
        )
        report_payload["export_paths"] = generate_all_formats(doc, Path(settings.REPORTS_DIR))
    except Exception:
        report_payload["export_paths"] = {}

    db.add(
        RuleEngineResult(
            scan_id=scan.id,
            payload=report.rules.model_dump() if report.rules else {},
            normalized_risk=report.rules.normalized_risk if report.rules else 0.0,
        )
    )
    if report.vision:
        db.add(
            ModelPrediction(
                scan_id=scan.id,
                model_name="vision",
                status=report.vision.status,
                payload=report.vision.model_dump(),
            )
        )
    if report.text:
        db.add(
            ModelPrediction(
                scan_id=scan.id,
                model_name="text",
                status=report.text.status,
                payload=report.text.model_dump(),
            )
        )
    if report.fusion:
        db.add(
            ModelPrediction(
                scan_id=scan.id,
                model_name="fusion",
                status=report.fusion.status,
                payload=report.fusion.model_dump(),
            )
        )

    for item in report.evidence:
        db.add(
            DetectedPattern(
                scan_id=scan.id,
                pattern_type=report.category.value,
                source=item.source,
                severity=item.severity,
                evidence=item.statement,
                meta=item.metadata,
            )
        )

    db.add(
        ScanResult(
            scan_id=scan.id,
            category=report.category.value,
            risk_score=report.risk_score,
            confidence=report.confidence,
            report=report_payload,
        )
    )

    scan.status = "completed"
    scan.completed_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(scan)
    return scan
