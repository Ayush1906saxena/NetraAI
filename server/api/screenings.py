"""Screening endpoints: create, upload fundus images, trigger analysis, get results.

Uses SQLAlchemy async sessions and the new InferenceService (EfficientNet-B3).
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
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


async def _get_screening_or_404(db: AsyncSession, screening_id: uuid.UUID) -> Screening:
    """Fetch a screening by ID or raise 404."""
    result = await db.execute(select(Screening).where(Screening.id == screening_id))
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    return screening


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/", response_model=ScreeningResponse, status_code=status.HTTP_201_CREATED)
async def create_screening(
    body: ScreeningCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new screening session."""
    screening = Screening(
        id=uuid.uuid4(),
        patient_id=body.patient_id,
        store_id=body.store_id,
        operator_id=uuid.UUID(current_user["user_id"]),
        status="created",
        notes=body.notes,
    )
    db.add(screening)
    await db.flush()
    await db.refresh(screening)
    logger.info("Screening created: %s by operator %s", screening.id, current_user["user_id"])
    return screening


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

    screening = await _get_screening_or_404(db, screening_id)

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
        from server.services.storage import upload_image
        await upload_image(
            image_bytes=image_bytes,
            bucket=settings.S3_BUCKET_IMAGES,
            key=s3_key,
            content_type=file.content_type or "image/jpeg",
        )
    except Exception as e:
        logger.warning("S3 upload failed (continuing without storage): %s", e)
        # In dev mode without MinIO, we still create the DB record

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
    screening.status = "images_uploaded"
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

    screening = await _get_screening_or_404(db, screening_id)

    if screening.status not in ("images_uploaded", "created"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot analyze screening in '{screening.status}' status",
        )

    # Get images for this screening
    img_result = await db.execute(
        select(Image).where(
            Image.screening_id == screening_id,
            Image.iqa_passed == True,
        )
    )
    images = img_result.scalars().all()
    if not images:
        raise HTTPException(status_code=400, detail="No quality-approved images found")

    screening.status = "analyzing"
    await db.flush()

    try:
        for img in images:
            # Try to get image bytes from S3
            img_bytes = None
            try:
                from server.services.storage import download_image
                img_bytes = await download_image(bucket=img.s3_bucket, key=img.s3_key)
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
        screening.referral_required = referral["referral_required"]
        screening.referral_urgency = referral["referral_urgency"]
        screening.referral_reason = referral["referral_reason"]
        screening.overall_risk = referral["overall_risk"]
        screening.raw_results = {"analysis_version": "v2", "model": "efficientnet_b3"}
        screening.status = "completed"
        screening.completed_at = datetime.now(timezone.utc)

        await db.flush()
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
        screening.status = "failed"
        await db.flush()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/{screening_id}", response_model=ScreeningResponse)
async def get_screening(
    screening_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get screening details and results."""
    screening = await _get_screening_or_404(db, screening_id)
    return screening


@router.get("/", response_model=ScreeningListResponse)
async def list_screenings(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    store_id: uuid.UUID | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List screenings with filtering and pagination."""
    query = select(Screening)

    if status_filter:
        query = query.where(Screening.status == status_filter)
    if store_id:
        query = query.where(Screening.store_id == store_id)

    # Non-admin users can only see their store's screenings
    if current_user["role"] not in ("admin", "doctor"):
        if current_user.get("store_id"):
            query = query.where(Screening.store_id == uuid.UUID(current_user["store_id"]))

    query = query.order_by(Screening.created_at.desc())

    # Count
    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar() or 0

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = result.scalars().all()

    return ScreeningListResponse(items=items, total=total, page=page, page_size=page_size)
