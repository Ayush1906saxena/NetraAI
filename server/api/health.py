"""Health and readiness check endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import settings
from server.dependencies import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Basic liveness probe."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def readiness_check(request: Request, db: AsyncSession = Depends(get_db)):
    """Deep readiness probe — checks DB, models, etc."""
    checks = {}

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error("Database readiness check failed: %s", e)
        checks["database"] = f"error: {str(e)}"

    # ML models check
    models = getattr(request.app.state, "models", None)
    if models is not None and models.is_loaded:
        checks["ml_models"] = "ok"
    else:
        checks["ml_models"] = "not loaded"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
