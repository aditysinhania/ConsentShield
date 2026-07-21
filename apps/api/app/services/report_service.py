"""Report export service — build documents and serve stored exports."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.report.document import ReportDocument
from ai.report.exporters import export_html, export_json, export_markdown, export_pdf
from ai.report.generator import generate_all_formats, generate_report_document

from app.core.config import settings
from app.models.scan import ScanResult, Screenshot, WebsiteScan


async def load_scan_report_context(
    db: AsyncSession,
    scan_id: uuid.UUID,
) -> tuple[WebsiteScan, dict[str, Any], str | None, str | None]:
    scan_row = await db.execute(select(WebsiteScan).where(WebsiteScan.id == scan_id))
    scan = scan_row.scalar_one_or_none()
    if scan is None:
        raise LookupError("Scan not found")

    result_row = await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))
    result = result_row.scalar_one_or_none()
    if result is None:
        raise LookupError("Report not found")

    shot_row = await db.execute(select(Screenshot).where(Screenshot.scan_id == scan_id))
    screenshot = shot_row.scalar_one_or_none()
    report = result.report or {}
    annotated = report.get("annotated_screenshot_path")
    return scan, {
        "risk_score": result.risk_score,
        "category": result.category,
        "confidence": result.confidence,
        "report": report,
    }, screenshot.storage_path if screenshot else None, annotated


def build_document_from_context(
    scan: WebsiteScan,
    ctx: dict[str, Any],
    screenshot_path: str | None,
    annotated_path: str | None,
) -> ReportDocument:
    return generate_report_document(
        scan_id=str(scan.id),
        url=scan.url,
        title=scan.title,
        risk_score=ctx["risk_score"],
        category=ctx["category"],
        confidence=ctx["confidence"],
        report=ctx["report"],
        screenshot_path=screenshot_path,
        annotated_screenshot_path=annotated_path,
        narrator=ctx["report"].get("narrator"),
    )


def save_report_exports(document: ReportDocument) -> dict[str, str]:
    out_dir = Path(settings.REPORTS_DIR)
    return generate_all_formats(document, out_dir)


def stored_export_path(scan_id: str, fmt: str) -> Path:
    ext = {"html": "html", "markdown": "md", "pdf": "pdf", "json": "json"}[fmt]
    return Path(settings.REPORTS_DIR) / f"{scan_id}.{ext}"
