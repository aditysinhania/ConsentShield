"""Model registry metadata endpoints."""

from fastapi import APIRouter

from app.core.config import settings
from app.services.scan_service import get_pipeline
from ai.registry.model_registry import ModelRegistry

router = APIRouter()


def _text_model_payload(registry: ModelRegistry) -> dict:
    ft = registry.finetuned_status()
    checkpoint = ft.get("checkpoint") or settings.MINILM_CHECKPOINT
    if ft.get("loaded"):
        return {
            "name": "Fine-tuned MiniLM",
            "status": "loaded",
            "checkpoint": checkpoint,
            "model_version": ft.get("model_version"),
            "device": ft.get("inference_device") or settings.DEVICE,
        }
    # Fall back to whatever text channel reports
    text = registry.text
    ready = text.is_ready()
    return {
        "name": "Fine-tuned MiniLM" if not ready else getattr(text, "name", "text_classifier"),
        "status": "loaded" if ready else "not_loaded",
        "checkpoint": checkpoint,
        "model_version": ft.get("model_version"),
        "device": ft.get("inference_device") or settings.DEVICE,
        "error": ft.get("error"),
    }


@router.get("")
@router.get("/")
async def models_status() -> dict:
    """Phase 2.5 model status (Fine-tuned MiniLM primary text model)."""
    pipe = get_pipeline()
    registry = getattr(pipe, "registry", None) or ModelRegistry.default()
    vision_ready = registry.vision.is_ready()
    return {
        "text_model": _text_model_payload(registry),
        "vision_model": {
            "name": "CLIP / Stub",
            "status": "loaded" if vision_ready else "not_loaded",
            "model": settings.VISION_MODEL_NAME,
        },
        "stub_mode": settings.AI_STUB_MODE,
        "phase4_backend": settings.PHASE4_BACKEND,
    }


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
            "minilm_checkpoint": settings.MINILM_CHECKPOINT,
        },
        "text_model": _text_model_payload(registry),
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
