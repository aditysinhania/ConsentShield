from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.detection import ModelPrediction
from app.models.scan import ScanResult, Screenshot
from app.schemas import ReportOut

router = APIRouter()


@router.get("/{scan_id}", response_model=ReportOut)
async def get_report(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> ReportOut:
    result = await db.execute(select(ScanResult).where(ScanResult.scan_id == scan_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")

    shot = await db.execute(select(Screenshot).where(Screenshot.scan_id == scan_id))
    screenshot = shot.scalar_one_or_none()
    report = row.report or {}

    return ReportOut(
        scan_id=scan_id,
        risk_score=row.risk_score,
        category=row.category,
        confidence=row.confidence,
        evidence=report.get("evidence", []),
        vision=report.get("vision"),
        text=report.get("text"),
        rules=report.get("rules"),
        fusion=report.get("fusion"),
        pipeline_notes=report.get("pipeline_notes", []),
        screenshot_path=screenshot.storage_path if screenshot else None,
    )
