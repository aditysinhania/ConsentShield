from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_optional_user
from app.db.session import get_db
from app.models.feedback import Feedback
from app.models.user import User
from app.schemas import FeedbackCreate, FeedbackOut

router = APIRouter()


@router.post("", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
async def create_feedback(
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Feedback:
    row = Feedback(
        user_id=user.id if user else None,
        scan_id=body.scan_id,
        label=body.label,
        comment=body.comment,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row
