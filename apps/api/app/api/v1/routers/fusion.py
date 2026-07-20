from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.detection import ModelPrediction

router = APIRouter()


@router.get("/{scan_id}")
async def fusion_for_scan(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(ModelPrediction).where(
            ModelPrediction.scan_id == scan_id, ModelPrediction.model_name == "fusion"
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Fusion prediction not found")
    return {"scan_id": str(scan_id), "status": row.status, "payload": row.payload}
