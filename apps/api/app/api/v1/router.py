from fastapi import APIRouter

from app.api.v1.routers import auth, feedback, fusion, models, report, rules, scan, text, vision

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(scan.router, prefix="/scan", tags=["scan"])
api_router.include_router(vision.router, prefix="/vision", tags=["vision"])
api_router.include_router(text.router, prefix="/text", tags=["text"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
api_router.include_router(fusion.router, prefix="/fusion", tags=["fusion"])
api_router.include_router(report.router, prefix="/report", tags=["report"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
