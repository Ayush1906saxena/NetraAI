"""Image endpoints: IQA check and Grad-CAM retrieval."""

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_current_user, get_db, get_models
from server.models.image import Image
from server.services.inference import ModelRegistry
from server.services.storage import generate_presigned_url

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/check-quality")
async def check_quality(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    models: ModelRegistry = Depends(get_models),
):
    """Run Image Quality Assessment on an uploaded fundus image."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    result = await models.check_quality(image_bytes)

    return {
        "filename": file.filename,
        "iqa_score": result["score"],
        "iqa_passed": result["score"] >= 0.5,
        "details": result.get("details", {}),
        "recommendations": _quality_recommendations(result),
    }


def _quality_recommendations(result: dict) -> list[str]:
    """Generate human-readable recommendations based on IQA results."""
    recs = []
    details = result.get("details", {})

    if result["score"] < 0.3:
        recs.append("Image quality is very poor. Please retake the image.")
    elif result["score"] < 0.5:
        recs.append("Image quality is below threshold. Consider retaking.")

    if details.get("blur_score", 1.0) < 0.4:
        recs.append("Image appears blurry — ensure camera focus is sharp.")
    if details.get("illumination_score", 1.0) < 0.4:
        recs.append("Illumination is uneven — adjust lighting.")
    if details.get("field_of_view_score", 1.0) < 0.4:
        recs.append("Field of view is insufficient — reposition the camera.")

    if not recs:
        recs.append("Image quality is acceptable.")

    return recs


@router.get("/{image_id}/gradcam")
async def get_gradcam(
    image_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the Grad-CAM visualization for an analyzed image."""
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if not image.gradcam_s3_key:
        raise HTTPException(status_code=404, detail="Grad-CAM not available for this image")

    url = await generate_presigned_url(
        bucket=image.s3_bucket,
        key=image.gradcam_s3_key,
    )

    return {
        "image_id": str(image.id),
        "eye": image.eye,
        "gradcam_url": url,
        "ai_results": image.ai_results,
    }
