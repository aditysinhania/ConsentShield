"""Model registry metadata endpoints."""

from fastapi import APIRouter

from app.core.config import settings
from app.services.scan_service import get_pipeline
from ai.registry.model_registry import ModelRegistry

router = APIRouter()


@router.get("/registry")
async def model_registry() -> dict:
    pipe = get_pipeline()
    registry = getattr(pipe, "registry", None) or ModelRegistry.default()
    return {
        "stub_mode": settings.AI_STUB_MODE,
        "ai_provider": settings.AI_PROVIDER,
        "phase4": {
            "text_model": settings.TEXT_MODEL_NAME,
            "vision_model": settings.VISION_MODEL_NAME,
            "device": settings.DEVICE,
            "backend": settings.PHASE4_BACKEND,
            "cache_models": settings.CACHE_MODELS,
            "models_root": settings.MODELS_ROOT,
        },
        "interfaces": registry.describe(),
        "modules": [
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
            {
                "name": pipe.text.name,
                "version": pipe.text.version,
                "ready": pipe.text.is_ready(),
            },
            {
                "name": pipe.vision.name,
                "version": pipe.vision.version,
                "ready": pipe.vision.is_ready(),
            },
        ],
    }
