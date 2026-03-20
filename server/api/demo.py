"""
Demo endpoint — single-image DR analysis without authentication.
Designed for quick demos and testing.

Usage:
    curl -X POST http://localhost:8000/v1/demo/analyze -F "file=@fundus.png"

Batch usage:
    curl -X POST http://localhost:8000/v1/demo/analyze-batch \
         -F "files=@left.png" -F "files=@right.png"
"""

import asyncio
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
MAX_BATCH_SIZE = 10  # Max files in a single batch request


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


@router.post("/analyze-batch")
async def demo_analyze_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Multiple fundus images (JPEG/PNG)"),
) -> dict[str, Any]:
    """
    Upload multiple fundus images and get DR analysis for all of them.

    No authentication required — intended for demo/testing.
    Processes all images concurrently using asyncio.gather.

    Returns a list of results in the same order as the uploaded files.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files ({len(files)}). Maximum is {MAX_BATCH_SIZE}.",
        )

    # Get the inference service
    inference_svc = getattr(request.app.state, "inference_service", None)
    if inference_svc is None or not inference_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DR model not loaded. The server is still starting or the checkpoint is missing.",
        )

    # Read all files first (validation pass)
    file_data: list[tuple[str, bytes, str]] = []
    for f in files:
        content_type = f.content_type or ""
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type for '{f.filename}': {content_type}. Use JPEG or PNG.",
            )

        image_bytes = await f.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty file: '{f.filename}'.",
            )
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{f.filename}' too large ({len(image_bytes)} bytes). Max {MAX_FILE_SIZE} bytes.",
            )
        file_data.append((f.filename or "unknown", image_bytes, content_type))

    # Process concurrently
    async def _analyze_one(filename: str, image_bytes: bytes) -> dict[str, Any]:
        """Analyze a single image, returning a result dict."""
        try:
            quality = await inference_svc.check_quality(image_bytes)
        except Exception as e:
            logger.error("Quality check error for %s: %s", filename, e)
            quality = {"score": 0.5, "passed": True, "details": {"error": str(e)}}

        try:
            result = await inference_svc.analyze_fundus(image_bytes)
        except Exception as e:
            logger.error("Inference error for %s: %s", filename, e, exc_info=True)
            return {
                "status": "error",
                "filename": filename,
                "error": str(e),
            }

        return {
            "status": "success",
            "filename": filename,
            "quality": quality,
            "analysis": result,
        }

    # Run all analyses concurrently
    tasks = [_analyze_one(fname, data) for fname, data, _ in file_data]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return {
        "status": "success",
        "total": len(results),
        "results": list(results),
    }
