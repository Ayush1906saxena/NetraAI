"""Screening endpoints: create, upload fundus images, trigger analysis, get results.

Uses the screening_service layer for CRUD operations and
InferenceService (EfficientNet-B3) for AI analysis.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import settings
from server.dependencies import get_current_user, get_db, get_inference_service
from server.models.image import Image
from server.models.screening import Screening
from server.schemas.screening import (
    AnalysisRequest,
    AnalysisResponse,
    ScreeningCreate,
    ScreeningListResponse,
    ScreeningResponse,
    UploadResponse,
)
from server.services.inference_v2 import InferenceService
from server.services import screening_service

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _compute_referral(screening: Screening) -> dict:
    """Compute referral recommendation based on DR grades on both eyes."""
    reasons = []
    urgency = "routine"

    for eye in ("left", "right"):
        dr_grade = getattr(screening, f"dr_grade_{eye}")
        if dr_grade is not None:
            if dr_grade >= 4:
                reasons.append(f"Proliferative DR in {eye} eye (grade {dr_grade})")
                urgency = "emergency"
            elif dr_grade == 3:
                reasons.append(f"Severe NPDR in {eye} eye (grade {dr_grade})")
                if urgency != "emergency":
                    urgency = "emergency"
            elif dr_grade == 2:
                reasons.append(f"Moderate NPDR in {eye} eye (grade {dr_grade})")
                if urgency not in ("emergency",):
                    urgency = "urgent"
            elif dr_grade == 1:
                reasons.append(f"Mild NPDR in {eye} eye (grade {dr_grade})")

        # Glaucoma / AMD from raw_results (future-proof)
        glaucoma_prob = getattr(screening, f"glaucoma_prob_{eye}", None)
        if glaucoma_prob is not None and glaucoma_prob > 0.5:
            reasons.append(f"Glaucoma risk in {eye} eye ({glaucoma_prob:.0%})")
            if urgency not in ("emergency",):
                urgency = "urgent"

        amd_prob = getattr(screening, f"amd_prob_{eye}", None)
        if amd_prob is not None and amd_prob > 0.5:
            reasons.append(f"AMD risk in {eye} eye ({amd_prob:.0%})")
            if urgency not in ("emergency",):
                urgency = "urgent"

    referral_required = len(reasons) > 0

    if urgency == "emergency":
        overall_risk = "urgent"
    elif urgency == "urgent":
        overall_risk = "high"
    elif reasons:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    return {
        "referral_required": referral_required,
        "referral_urgency": urgency if referral_required else None,
        "referral_reason": "; ".join(reasons) if reasons else None,
        "overall_risk": overall_risk,
    }


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/", response_model=ScreeningResponse, status_code=status.HTTP_201_CREATED)
async def create_screening_endpoint(
    body: ScreeningCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new screening session."""
    new_screening = await screening_service.create_screening(
        db=db,
        patient_id=body.patient_id,
        store_id=body.store_id,
        operator_id=uuid.UUID(current_user["user_id"]),
        notes=body.notes,
    )
    logger.info("Screening created: %s by operator %s", new_screening.id, current_user["user_id"])
    return new_screening


@router.post("/{screening_id}/upload/{eye}", response_model=UploadResponse)
async def upload_fundus_image(
    screening_id: uuid.UUID,
    eye: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    inference: InferenceService = Depends(get_inference_service),
):
    """Upload a fundus image, run quality check, store metadata in DB."""
    if eye not in ("left", "right"):
        raise HTTPException(status_code=400, detail="Eye must be 'left' or 'right'")

    screening = await screening_service.get_screening(db, screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Read file bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Run quality check using the new inference service
    iqa_result = await inference.check_quality(image_bytes)
    iqa_score = iqa_result["score"]
    iqa_passed = iqa_result["passed"]

    # Try to upload to S3/MinIO (graceful failure for dev without MinIO)
    s3_key = f"screenings/{screening_id}/{eye}/{file.filename}"
    try:
        from server.services.storage import upload_fundus_image as upload_to_minio
        s3_key = await upload_to_minio(
            screening_id=str(screening_id),
            eye=eye,
            image_bytes=image_bytes,
        )
    except Exception as e:
        logger.warning("S3 upload failed (continuing without storage): %s", e)

    # Create image record
    image = Image(
        id=uuid.uuid4(),
        screening_id=screening_id,
        eye=eye,
        image_type="fundus",
        s3_key=s3_key,
        s3_bucket=settings.S3_BUCKET_IMAGES,
        filename=file.filename or "fundus.jpg",
        content_type=file.content_type or "image/jpeg",
        file_size_bytes=len(image_bytes),
        iqa_score=iqa_score,
        iqa_passed=iqa_passed,
        iqa_details=iqa_result.get("details"),
    )
    db.add(image)

    # Update screening status
    await screening_service.update_screening_status(db, screening_id, "images_uploaded")
    await db.flush()
    await db.refresh(image)

    logger.info(
        "Image uploaded for screening %s, eye=%s, IQA=%.2f passed=%s",
        screening_id, eye, iqa_score, iqa_passed,
    )

    return UploadResponse(
        image_id=image.id,
        eye=eye,
        iqa_score=iqa_score,
        iqa_passed=iqa_passed,
        message="Image uploaded successfully" if iqa_passed else "Image quality is poor — consider retaking",
    )


@router.post("/{screening_id}/analyze", response_model=AnalysisResponse)
async def trigger_analysis(
    screening_id: uuid.UUID,
    body: AnalysisRequest | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    inference: InferenceService = Depends(get_inference_service),
):
    """Trigger AI analysis on uploaded fundus images for a screening."""
    if body is None:
        body = AnalysisRequest()

    screening = await screening_service.get_screening(db, screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening not found")

    if screening.status not in ("images_uploaded", "created"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot analyze screening in '{screening.status}' status",
        )

    # Get images for this screening
    from sqlalchemy import select
    from server.models.image import Image

    img_result = await db.execute(
        select(Image).where(
            Image.screening_id == screening_id,
            Image.iqa_passed == True,
        )
    )
    images = img_result.scalars().all()
    if not images:
        raise HTTPException(status_code=400, detail="No quality-approved images found")

    await screening_service.update_screening_status(db, screening_id, "analyzing")

    try:
        for img in images:
            # Try to get image bytes from S3
            img_bytes = None
            try:
                from server.services.storage import download_fundus_image
                img_bytes = await download_fundus_image(s3_key=img.s3_key)
            except Exception as e:
                logger.warning("Could not download image from S3: %s", e)

            if img_bytes is None:
                logger.warning("Skipping image %s — could not retrieve bytes", img.id)
                continue

            # Run DR analysis using the new inference service
            analysis = await inference.analyze_fundus(img_bytes)

            # Store full results as JSON
            img.ai_results = analysis

            eye = img.eye
            dr_info = analysis.get("dr", {})
            setattr(screening, f"dr_grade_{eye}", dr_info.get("grade"))
            setattr(screening, f"dr_confidence_{eye}", dr_info.get("confidence"))

        # Compute referral
        referral = _compute_referral(screening)

        # Save results via service
        await screening_service.save_analysis_results(
            db=db,
            screening_id=screening_id,
            results={
                "referral_required": referral["referral_required"],
                "referral_urgency": referral["referral_urgency"],
                "referral_reason": referral["referral_reason"],
                "overall_risk": referral["overall_risk"],
                "dr_grade_left": screening.dr_grade_left,
                "dr_grade_right": screening.dr_grade_right,
                "dr_confidence_left": screening.dr_confidence_left,
                "dr_confidence_right": screening.dr_confidence_right,
                "raw_results": {"analysis_version": "v2", "model": "efficientnet_b3"},
            },
        )

        await db.refresh(screening)

        logger.info("Screening %s analysis completed. Risk: %s", screening_id, screening.overall_risk)
        return AnalysisResponse(
            screening_id=screening_id,
            status="completed",
            message=f"Analysis complete. Overall risk: {screening.overall_risk}",
            task_id=None,
        )

    except Exception as e:
        logger.error("Analysis failed for screening %s: %s", screening_id, e, exc_info=True)
        await screening_service.update_screening_status(db, screening_id, "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/{screening_id}", response_model=ScreeningResponse)
async def get_screening_endpoint(
    screening_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get screening details and results."""
    screening = await screening_service.get_screening(db, screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening not found")
    return screening


@router.get("/", response_model=ScreeningListResponse)
async def list_screenings_endpoint(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    store_id: uuid.UUID | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List screenings with filtering and pagination."""
    # Non-admin users can only see their store's screenings
    effective_store_id = store_id
    if current_user["role"] not in ("admin", "doctor"):
        if current_user.get("store_id"):
            effective_store_id = uuid.UUID(current_user["store_id"])

    skip = (page - 1) * page_size
    items = await screening_service.list_screenings(
        db=db,
        skip=skip,
        limit=page_size,
        store_id=effective_store_id,
        status=status_filter,
    )
    total = await screening_service.count_screenings(
        db=db,
        store_id=effective_store_id,
        status=status_filter,
    )

    return ScreeningListResponse(items=items, total=total, page=page, page_size=page_size)
