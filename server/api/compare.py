"""
Compare Eyes endpoint — analyzes left and right eye images and flags asymmetry.

Usage:
    curl -X POST http://localhost:8000/v1/demo/compare-eyes \
         -F "left_eye=@left.png" -F "right_eye=@right.png"
"""

import logging
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

# Asymmetry thresholds
GRADE_DIFFERENCE_THRESHOLD = 2
CDR_DIFFERENCE_THRESHOLD = 0.1


def _validate_upload(file: UploadFile, label: str) -> None:
    """Validate content type of an uploaded file."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type for {label}: {content_type}. Use JPEG or PNG.",
        )


async def _read_and_validate(file: UploadFile, label: str) -> bytes:
    """Read file bytes and validate size."""
    _validate_upload(file, label)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Empty file uploaded for {label}.",
        )
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{label} file too large ({len(image_bytes)} bytes). Max {MAX_FILE_SIZE} bytes.",
        )
    return image_bytes


async def _analyze_single(
    inference_svc: Any,
    image_bytes: bytes,
    label: str,
    use_tta: Optional[bool],
    high_confidence_mode: bool,
) -> dict[str, Any]:
    """Run full analysis on a single eye image."""
    try:
        quality = await inference_svc.check_quality(image_bytes)
    except Exception as e:
        logger.error("Quality check error for %s: %s", label, e)
        quality = {"score": 0.5, "passed": True, "details": {"error": str(e)}}

    try:
        result = await inference_svc.analyze_fundus(
            image_bytes,
            use_tta=use_tta,
            high_confidence_mode=high_confidence_mode,
        )
    except Exception as e:
        logger.error("Inference error for %s: %s", label, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed for {label}: {str(e)}",
        )

    return {
        "status": result.get("status", "success"),
        "quality": quality,
        "analysis": result.get("analysis", {}),
        "referral": result.get("referral", {}),
        "gradcam": result.get("gradcam", {}),
        "model_info": result.get("model_info", {}),
        "meta": result.get("meta", {}),
    }


def _extract_grade(result: dict[str, Any]) -> int:
    """Extract DR grade from analysis result."""
    analysis = result.get("analysis", {})
    dr = analysis.get("dr", {})
    return dr.get("grade", 0)


def _extract_confidence(result: dict[str, Any]) -> float:
    """Extract confidence from analysis result."""
    analysis = result.get("analysis", {})
    dr = analysis.get("dr", {})
    return dr.get("confidence", 0.0)


def _extract_cdr(result: dict[str, Any]) -> Optional[float]:
    """Extract cup-to-disc ratio from glaucoma analysis if available."""
    analysis = result.get("analysis", {})
    glaucoma = analysis.get("glaucoma", {})
    cdr = glaucoma.get("cdr", None)
    return cdr


GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def _compare_results(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare left and right eye results and flag asymmetry."""
    left_grade = _extract_grade(left_result)
    right_grade = _extract_grade(right_result)
    left_confidence = _extract_confidence(left_result)
    right_confidence = _extract_confidence(right_result)
    left_cdr = _extract_cdr(left_result)
    right_cdr = _extract_cdr(right_result)

    grade_difference = abs(left_grade - right_grade)
    asymmetry_details: list[str] = []

    # Check grade asymmetry
    grade_asymmetry = grade_difference >= GRADE_DIFFERENCE_THRESHOLD
    if grade_asymmetry:
        asymmetry_details.append(
            f"DR grade difference of {grade_difference} between eyes "
            f"(Left: {GRADE_NAMES.get(left_grade, 'Unknown')}, "
            f"Right: {GRADE_NAMES.get(right_grade, 'Unknown')})"
        )

    # Check CDR asymmetry
    cdr_difference: Optional[float] = None
    cdr_asymmetry = False
    if left_cdr is not None and right_cdr is not None:
        cdr_difference = abs(left_cdr - right_cdr)
        cdr_asymmetry = cdr_difference >= CDR_DIFFERENCE_THRESHOLD
        if cdr_asymmetry:
            asymmetry_details.append(
                f"Cup-to-disc ratio difference of {cdr_difference:.2f} "
                f"(Left: {left_cdr:.2f}, Right: {right_cdr:.2f}) — "
                f"may indicate asymmetric glaucoma risk"
            )

    asymmetry_flag = grade_asymmetry or cdr_asymmetry

    # Determine worse eye
    worse_eye = "equal"
    if left_grade > right_grade:
        worse_eye = "left"
    elif right_grade > left_grade:
        worse_eye = "right"
    elif left_confidence < right_confidence:
        # Same grade — worse eye is the one with lower confidence in healthy
        worse_eye = "left" if left_grade > 0 else "equal"

    # Clinical significance
    if grade_difference == 0:
        clinical_significance = "Both eyes show the same DR grade. Symmetric findings."
    elif grade_difference == 1:
        clinical_significance = (
            "Minor difference between eyes (1 grade). This is common and may "
            "reflect normal variation. Monitor the worse eye more closely."
        )
    elif grade_difference >= 2:
        clinical_significance = (
            f"Significant asymmetry detected ({grade_difference} grade difference). "
            "Asymmetric diabetic retinopathy may indicate unilateral vascular "
            "compromise or other pathology. Urgent ophthalmologist review recommended."
        )
    else:
        clinical_significance = "Unable to determine clinical significance."

    if cdr_asymmetry and cdr_difference is not None:
        clinical_significance += (
            f" Additionally, CDR asymmetry of {cdr_difference:.2f} detected — "
            "this is a risk factor for glaucoma and warrants further evaluation."
        )

    return {
        "grade_difference": grade_difference,
        "cdr_difference": round(cdr_difference, 3) if cdr_difference is not None else None,
        "asymmetry_flag": asymmetry_flag,
        "asymmetry_details": asymmetry_details,
        "worse_eye": worse_eye,
        "clinical_significance": clinical_significance,
        "left_grade": left_grade,
        "left_grade_name": GRADE_NAMES.get(left_grade, "Unknown"),
        "right_grade": right_grade,
        "right_grade_name": GRADE_NAMES.get(right_grade, "Unknown"),
    }


@router.post("/compare-eyes")
async def compare_eyes(
    request: Request,
    left_eye: UploadFile = File(..., description="Left eye fundus image (JPEG/PNG)"),
    right_eye: UploadFile = File(..., description="Right eye fundus image (JPEG/PNG)"),
    mode: Optional[str] = Query(
        None,
        description="Inference mode: 'fast' (default), 'tta', 'high_confidence'.",
    ),
) -> dict[str, Any]:
    """
    Upload left and right eye fundus images for comparative analysis.

    Runs DR analysis on both eyes, compares grades and CDR,
    and flags any significant asymmetry.

    Returns:
    - left_results: full analysis for left eye
    - right_results: full analysis for right eye
    - comparison: grade_difference, asymmetry_flag, asymmetry_details,
                  worse_eye, clinical_significance
    """
    # Get inference service
    inference_svc = getattr(request.app.state, "inference_service", None)
    if inference_svc is None or not inference_svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DR model not loaded. The server is still starting or the checkpoint is missing.",
        )

    # Resolve mode
    use_tta = None
    high_confidence_mode = False
    if mode == "tta":
        use_tta = True
    elif mode == "high_confidence":
        high_confidence_mode = True

    # Read and validate both files
    left_bytes = await _read_and_validate(left_eye, "left_eye")
    right_bytes = await _read_and_validate(right_eye, "right_eye")

    # Analyze both eyes
    import asyncio

    left_task = _analyze_single(
        inference_svc, left_bytes, "left_eye", use_tta, high_confidence_mode
    )
    right_task = _analyze_single(
        inference_svc, right_bytes, "right_eye", use_tta, high_confidence_mode
    )

    left_result, right_result = await asyncio.gather(left_task, right_task)

    # Compare results
    comparison = _compare_results(left_result, right_result)

    return {
        "status": "success",
        "left_filename": left_eye.filename,
        "right_filename": right_eye.filename,
        "left_results": left_result,
        "right_results": right_result,
        "comparison": comparison,
    }
