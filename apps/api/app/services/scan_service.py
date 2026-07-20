"""Scan processing service — orchestrates AI pipeline and persistence."""

from __future__ import annotations

import base64
import sys
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

from app.core.config import settings
from app.models.detection import DetectedPattern, ModelPrediction, RuleEngineResult
from app.models.scan import ScanResult, Screenshot, WebsiteScan

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline


def _save_screenshot(scan_id: uuid.UUID, b64: str | None) -> str | None:
    if not b64:
        return None
    raw = b64.split(",", 1)[-1] if "," in b64 else b64
    data = base64.b64decode(raw)
    out_dir = Path(settings.SCREENSHOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scan_id}.png"
    path.write_bytes(data)
    return str(path)


async def create_and_process_scan(
    db: AsyncSession,
    *,
    payload: ScanPayload,
    user_id: uuid.UUID | None,
) -> WebsiteScan:
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

    shot_path = _save_screenshot(scan.id, payload.screenshot_base64)
    if shot_path:
        db.add(Screenshot(scan_id=scan.id, storage_path=shot_path))

    report = get_pipeline().run(payload)

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
            report=report.model_dump(),
        )
    )

    scan.status = "completed"
    scan.completed_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(scan)
    return scan
