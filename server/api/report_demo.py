"""
Demo endpoint — generate a PDF screening report from a single fundus image.

Usage:
    curl -X POST http://localhost:8000/v1/demo/report \
         -F "file=@fundus.png" \
         -G -d "patient_name=John Doe" -d "patient_age=55"
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response

from server.services.report_gen import ReportGenerator

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

# Singleton report generator
_report_gen = ReportGenerator()


@router.post("/report")
async def demo_report(
    request: Request,
    file: UploadFile = File(..., description="Fundus image (JPEG/PNG)"),
    patient_name: str = Query(default="Demo Patient", description="Patient name"),
    patient_age: Optional[int] = Query(default=None, description="Patient age"),
    patient_gender: Optional[str] = Query(default=None, description="Patient gender (M/F/Other)"),
) -> Response:
    """
    Upload a fundus image and receive a full PDF screening report.

    Runs DR inference with the trained model, generates GradCAM,
    and produces a formatted PDF report (or HTML if WeasyPrint is not installed).

    No authentication required — intended for demo/testing.
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

    # Get inference service
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

    # Generate report
    try:
        output_bytes, filename, report_content_type = await _report_gen.generate_from_inference(
            inference_result=result,
            fundus_image_bytes=image_bytes,
            quality_result=quality,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
        )
    except Exception as e:
        logger.error("Report generation error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )

    return Response(
        content=output_bytes,
        media_type=report_content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
