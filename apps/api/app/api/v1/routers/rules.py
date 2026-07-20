from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.detection import ModelPrediction, RuleEngineResult
from app.services.scan_service import get_pipeline

router = APIRouter()


@router.get("/catalog")
async def rule_catalog() -> dict:
    engine = get_pipeline().rules
    return {"rules": engine.list_rules(), "count": len(engine.list_rules())}


@router.get("/{scan_id}")
async def rule_results(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(RuleEngineResult).where(RuleEngineResult.scan_id == scan_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Rule results not found")
    return {"scan_id": str(scan_id), "normalized_risk": row.normalized_risk, "payload": row.payload}
