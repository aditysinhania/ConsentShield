from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.common.types import ScanPayload

from app.api.deps import get_optional_user
from app.db.session import get_db
from app.models.scan import WebsiteScan
from app.models.user import User
from app.schemas import ScanCreate, ScanOut
from app.services.scan_service import create_and_process_scan

router = APIRouter()


@router.post("", response_model=ScanOut, status_code=201)
async def create_scan(
    body: ScanCreate,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> WebsiteScan:
    payload = ScanPayload(
        url=body.url,
        title=body.title,
        html=body.html,
        css_snapshot=body.css_snapshot,
        visible_text=body.visible_text,
        screenshot_base64=body.screenshot_base64,
        viewport=body.viewport,
        scroll_position=body.scroll_position,
        collected_at=body.collected_at,
    )
    return await create_and_process_scan(
        db, payload=payload, user_id=user.id if user else None
    )


@router.get("", response_model=list[ScanOut])
async def list_scans(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_user),
    limit: int = 50,
) -> list[WebsiteScan]:
    stmt = select(WebsiteScan).order_by(WebsiteScan.created_at.desc()).limit(limit)
    if user:
        stmt = stmt.where(WebsiteScan.user_id == user.id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{scan_id}", response_model=ScanOut)
async def get_scan(scan_id: UUID, db: AsyncSession = Depends(get_db)) -> WebsiteScan:
    result = await db.execute(select(WebsiteScan).where(WebsiteScan.id == scan_id))
    scan = result.scalar_one_or_none()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan
