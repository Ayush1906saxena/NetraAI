"""
Model information endpoint — returns loaded models, versions, sizes, and inference stats.

Useful for monitoring and debugging deployed ML models.

Usage:
    GET http://localhost:8000/v1/models/info
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/info")
async def get_models_info(request: Request) -> dict[str, Any]:
    """
    Returns information about all loaded ML models.

    Includes:
    - List of loaded models with name, architecture, checkpoint path, parameter count
    - Device information
    - Ensemble size
    - Temperature scaling value
    - Inference metrics (request counts, latency, cache stats)
    """
    inference_svc = getattr(request.app.state, "inference_service", None)
    if inference_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized.",
        )

    try:
        info = inference_svc.get_models_info()
    except Exception as e:
        logger.error("Failed to get models info: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}",
        )

    return {
        "status": "success",
        "data": info,
    }
