"""
Comprehensive eye health screening endpoint.

POST /v1/eye-health/screen — upload a fundus image and receive a combined report
covering: image quality, DR grading, multi-disease classification, and glaucoma CDR.

Accepts optional patient metadata (age, sex, diabetes duration, hypertension status)
to enrich the assessment.
"""

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

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


def _risk_level_rank(level: str) -> int:
    return {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}.get(level, 0)


def _build_recommendations(
    dr_result: dict | None,
    multi_disease_result: dict | None,
    glaucoma_result: dict | None,
    patient_age: int | None,
    has_hypertension: bool | None,
) -> list[str]:
    """Generate personalized clinical recommendations based on all findings."""
    recs: list[str] = []

    # DR recommendations
    if dr_result:
        grade = dr_result.get("grade", 0)
        if grade >= 3:
            recs.append("Urgent referral to a retina specialist for diabetic retinopathy management.")
        elif grade == 2:
            recs.append("Schedule follow-up with an ophthalmologist within 3-6 months for DR monitoring.")
        elif grade == 1:
            recs.append("Annual diabetic eye screening recommended; optimize blood sugar control.")

    # Multi-disease recommendations
    if multi_disease_result:
        for flag in multi_disease_result.get("critical_flags", []):
            recs.append(f"{flag['condition']}: {flag['what_to_do']}")

        detected = multi_disease_result.get("detected_conditions", [])
        if "Cataract" in detected:
            recs.append("Consult an ophthalmologist for cataract evaluation and surgical planning if vision is impaired.")
        if "Pathological Myopia" in detected:
            recs.append("Regular retinal monitoring recommended for pathological myopia-related complications.")

    # Glaucoma CDR recommendations
    if glaucoma_result:
        cdr = glaucoma_result.get("cdr")
        if cdr is not None and cdr > 0.6:
            recs.append("Elevated cup-to-disc ratio detected; refer for glaucoma workup including IOP and visual fields.")

    # Age-based
    if patient_age and patient_age >= 60:
        recs.append("Annual comprehensive eye examination recommended for patients over 60.")

    # Hypertension
    if has_hypertension:
        recs.append("Regular fundus screening recommended due to hypertension; monitor for hypertensive retinopathy.")

    # Default if nothing flagged
    if not recs:
        recs.append("No urgent findings. Continue routine eye examinations per your provider's schedule.")

    return recs


@router.post("/screen")
async def comprehensive_screen(
    request: Request,
    file: UploadFile = File(..., description="Fundus image (JPEG/PNG)"),
    patient_age: Optional[int] = Query(None, description="Patient age in years"),
    patient_sex: Optional[str] = Query(None, description="Patient sex (Male/Female)"),
    diabetes_duration: Optional[float] = Query(None, description="Years since diabetes diagnosis"),
    has_hypertension: Optional[bool] = Query(None, description="Whether patient has hypertension"),
):
    """
    Comprehensive eye health screening.

    Runs all available AI models on the uploaded fundus image:
    - Image Quality Assessment (IQA)
    - Diabetic Retinopathy grading (5-class)
    - Multi-disease classification (8 conditions)
    - Glaucoma cup-to-disc ratio estimation

    Returns a unified screening report with risk assessment and recommendations.
    """
    # ── Validate file ────────────────────────────────────────────────────
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(image_bytes) / (1024*1024):.1f} MB). Max is {MAX_FILE_SIZE / (1024*1024):.0f} MB.",
        )
    if len(image_bytes) < 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small or possibly corrupt.",
        )

    screening_id = str(uuid.uuid4())

    # ── Get inference service ────────────────────────────────────────────
    inference_svc = getattr(request.app.state, "inference_service", None)

    # ── Run DR grading ───────────────────────────────────────────────────
    dr_result = None
    if inference_svc and inference_svc.is_loaded:
        try:
            dr_raw = await inference_svc.analyze_fundus(image_bytes)
            dr_result = {
                "grade": dr_raw.get("grade", 0),
                "grade_name": dr_raw.get("grade_name", "Unknown"),
                "confidence": dr_raw.get("confidence", 0.0),
                "class_probabilities": dr_raw.get("class_probabilities", {}),
                "is_referable": dr_raw.get("is_referable", False),
            }
        except Exception as e:
            logger.warning("DR grading failed: %s", e)
            dr_result = {"error": str(e)}

    # ── Run IQA ──────────────────────────────────────────────────────────
    quality_result = None
    if inference_svc and inference_svc.iqa_loaded:
        try:
            iqa_raw = await inference_svc.check_quality(image_bytes)
            quality_result = iqa_raw
        except Exception as e:
            logger.warning("IQA check failed: %s", e)
            quality_result = {"error": str(e)}

    # ── Run multi-disease classification ─────────────────────────────────
    multi_disease_result = None
    multi_disease_svc = getattr(request.app.state, "multi_disease_service", None)
    if multi_disease_svc and multi_disease_svc.is_loaded:
        try:
            multi_disease_result = multi_disease_svc.predict(image_bytes)
        except Exception as e:
            logger.warning("Multi-disease classification failed: %s", e)
            multi_disease_result = {"error": str(e)}

    # ── Run glaucoma CDR ─────────────────────────────────────────────────
    glaucoma_result = None
    if inference_svc and inference_svc.glaucoma_loaded:
        try:
            glaucoma_raw = await inference_svc.analyze_glaucoma(image_bytes)
            glaucoma_result = glaucoma_raw
        except Exception as e:
            logger.warning("Glaucoma analysis failed: %s", e)
            glaucoma_result = {"error": str(e)}

    # ── Determine overall risk ───────────────────────────────────────────
    risk_levels = []

    if dr_result and "grade" in dr_result:
        grade = dr_result["grade"]
        if grade >= 3:
            risk_levels.append("high")
        elif grade == 2:
            risk_levels.append("moderate")
        elif grade == 1:
            risk_levels.append("low")
        else:
            risk_levels.append("none")

    if multi_disease_result and "overall_risk" in multi_disease_result:
        risk_levels.append(multi_disease_result["overall_risk"])

    if glaucoma_result and "cdr" in glaucoma_result:
        cdr = glaucoma_result.get("cdr", 0)
        if cdr > 0.7:
            risk_levels.append("high")
        elif cdr > 0.5:
            risk_levels.append("moderate")

    overall_risk = "none"
    for r in risk_levels:
        if _risk_level_rank(r) > _risk_level_rank(overall_risk):
            overall_risk = r

    # ── Count conditions ─────────────────────────────────────────────────
    conditions_found = 0
    conditions_screened = 8
    if multi_disease_result and "detected_conditions" in multi_disease_result:
        conditions_found = len(multi_disease_result["detected_conditions"])

    # ── Build recommendations ────────────────────────────────────────────
    recommendations = _build_recommendations(
        dr_result, multi_disease_result, glaucoma_result,
        patient_age, has_hypertension,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    summary_parts = []
    if multi_disease_result and "summary" in multi_disease_result:
        summary_parts.append(multi_disease_result["summary"])
    if dr_result and "grade_name" in dr_result:
        summary_parts.append(f"DR grading: {dr_result['grade_name']}.")
    if not summary_parts:
        summary_parts.append("Screening completed. Some models were unavailable.")

    return {
        "screening_id": screening_id,
        "quality": quality_result,
        "dr_grading": dr_result,
        "multi_disease": {
            "conditions": multi_disease_result.get("conditions", []) if multi_disease_result else [],
            "detected_conditions": multi_disease_result.get("detected_conditions", []) if multi_disease_result else [],
            "critical_flags": multi_disease_result.get("critical_flags", []) if multi_disease_result else [],
        } if multi_disease_result else None,
        "glaucoma": glaucoma_result,
        "overall_assessment": {
            "risk_level": overall_risk,
            "conditions_found": conditions_found,
            "conditions_screened": conditions_screened,
            "summary": " ".join(summary_parts),
            "recommendations": recommendations,
        },
        "patient_info": {
            "age": patient_age,
            "sex": patient_sex,
            "diabetes_duration_years": diabetes_duration,
            "has_hypertension": has_hypertension,
        },
    }
