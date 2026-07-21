from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.report.exporters import export_html, export_markdown, export_pdf

from app.db.session import get_db
from app.models.scan import ScanResult, Screenshot
from app.schemas import ReportOut
from app.services.report_service import (
    build_document_from_context,
    load_scan_report_context,
    stored_export_path,
)

router = APIRouter()


def _report_out(
    scan_id: UUID,
    row: ScanResult,
    report: dict,
    screenshot_path: str | None,
) -> ReportOut:
    return ReportOut(
        scan_id=scan_id,
        risk_score=row.risk_score,
        category=row.category,
        confidence=row.confidence,
        confidence_breakdown=report.get("confidence_breakdown"),
        evidence=report.get("evidence", []),
        rule_traces=report.get("rule_traces", []),
        severity=report.get("severity"),
        pattern_clusters=report.get("pattern_clusters", []),
        accessibility=report.get("accessibility"),
        timeline=report.get("timeline", []),
        performance=report.get("performance"),
        vision=report.get("vision"),
        text=report.get("text"),
        rules=report.get("rules"),
        fusion=report.get("fusion"),
        pipeline_notes=report.get("pipeline_notes", []),
        screenshot_path=screenshot_path,
        annotated_screenshot_path=report.get("annotated_screenshot_path"),
        export_paths=report.get("export_paths"),
    )


async def _document_or_404(db: AsyncSession, scan_id: UUID):
    try:
        scan, ctx, shot, ann = await load_scan_report_context(db, scan_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return build_document_from_context(scan, ctx, shot, ann)


@router.get("/{scan_id}/html", response_class=HTMLResponse)
async def get_report_html(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> HTMLResponse:
    path = stored_export_path(str(scan_id), "html")
    if path.is_file():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    document = await _document_or_404(db, scan_id)
    return HTMLResponse(export_html(document))


@router.get("/{scan_id}/markdown", response_class=PlainTextResponse)
async def get_report_markdown(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> PlainTextResponse:
    path = stored_export_path(str(scan_id), "markdown")
    if path.is_file():
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")
    document = await _document_or_404(db, scan_id)
    return PlainTextResponse(export_markdown(document), media_type="text/markdown")


@router.get("/{scan_id}/pdf")
async def get_report_pdf(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> Response:
    path = stored_export_path(str(scan_id), "pdf")
    if path.is_file():
        return Response(path.read_bytes(), media_type="application/pdf")
    document = await _document_or_404(db, scan_id)
    return Response(export_pdf(document), media_type="application/pdf")


@router.get("/{scan_id}", response_model=ReportOut)
async def get_report(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> ReportOut:
    result = await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")

    shot = await db.execute(select(Screenshot).where(Screenshot.scan_id == scan_id))
    screenshot = shot.scalar_one_or_none()
    report = row.report or {}

    return _report_out(
        scan_id,
        row,
        report,
        screenshot.storage_path if screenshot else None,
    )
