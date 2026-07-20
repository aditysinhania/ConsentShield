"""Model registry metadata endpoints."""

from fastapi import APIRouter

from app.core.config import settings
from app.services.scan_service import get_pipeline

router = APIRouter()


@router.get("/registry")
async def model_registry() -> dict:
    pipe = get_pipeline()
    return {
        "stub_mode": settings.AI_STUB_MODE,
        "modules": [
            {
                "name": pipe.vision.name,
                "version": pipe.vision.version,
                "ready": pipe.vision.is_ready(),
                "path": settings.VISION_MODEL_PATH or None,
            },
            {
                "name": pipe.text.name,
                "version": pipe.text.version,
                "ready": pipe.text.is_ready(),
                "path": settings.TEXT_MODEL_PATH or None,
            },
            {
                "name": pipe.rules.name,
                "version": pipe.rules.version,
                "ready": pipe.rules.is_ready(),
                "rules": len(pipe.rules.list_rules()),
            },
            {
                "name": pipe.fusion.name,
                "version": pipe.fusion.version,
                "ready": pipe.fusion.is_ready(),
            },
            {
                "name": pipe.explainer.name,
                "version": pipe.explainer.version,
                "ready": pipe.explainer.is_ready(),
            },
        ],
    }
