"""
Demo endpoint — single-image DR analysis without authentication.
Designed for quick demos and testing.

Usage:
    curl -X POST http://localhost:8000/v1/demo/analyze -F "file=@fundus.png"
"""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/analyze")
async def demo_analyze(
    request: Request,
    file: UploadFile = File(..., description="Fundus image (JPEG/PNG)"),
) -> dict[str, Any]:
    """
    Upload a single fundus image and get full DR analysis.

    No authentication required — intended for demo/testing.

    Returns DR grade, probabilities, confidence, GradCAM heatmap (base64),
    and referral recommendation.
    """
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type}. Use JPEG or PNG.",
        )

    # Read file bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(image_bytes)} bytes). Max {MAX_FILE_SIZE} bytes.",
        )

    # Get the inference service from app state
    inference_svc = getattr(request.app.state, "inference_service", None)
    if inference_svc is None or not inference_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DR model not loaded. The server is still starting or the checkpoint is missing.",
        )

    # Run quality check
    try:
        quality = await inference_svc.check_quality(image_bytes)
    except Exception as e:
        logger.error("Quality check error: %s", e)
        quality = {"score": 0.5, "passed": True, "details": {"error": str(e)}}

    # Run DR analysis
    try:
        result = await inference_svc.analyze_fundus(image_bytes)
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    return {
        "status": "success",
        "filename": file.filename,
        "quality": quality,
        "analysis": result,
    }
