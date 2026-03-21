"""
Multi-Condition Screening Service.

Screens for additional ocular conditions beyond diabetic retinopathy using
heuristic-based rules applied to the fundus analysis results. This is NOT
a trained classifier for these conditions -- it uses clinical heuristics
and correlations to flag potential concerns for further investigation.

Conditions screened:
  1. Hypertensive Retinopathy
  2. Age-related Macular Degeneration (AMD)
  3. Cataract Indicators
  4. Diabetic Macular Edema (DME) Risk
"""

from typing import Any, Optional


RISK_LEVELS = ["none", "low", "moderate", "high"]


def _clamp_risk(level: str) -> str:
    """Ensure risk level is one of the valid values."""
    return level if level in RISK_LEVELS else "none"


def _screen_hypertensive_retinopathy(
    dr_grade: int,
    has_hypertension: bool,
    glaucoma_cdr: Optional[float] = None,
) -> dict[str, Any]:
    """
    Screen for hypertensive retinopathy based on DR findings and patient history.

    Heuristic: DR signs (hemorrhages, exudates) combined with hypertension
    history suggest concurrent hypertensive retinopathy. High CDR may also
    indicate vascular damage.
    """
    risk_level = "none"
    findings: list[str] = []

    if has_hypertension:
        findings.append("Patient has reported hypertension history")
        if dr_grade >= 2:
            risk_level = "high"
            findings.append(
                "DR grade >= 2 with hypertension suggests likely concurrent "
                "hypertensive retinopathy (shared vascular pathology)"
            )
        elif dr_grade == 1:
            risk_level = "moderate"
            findings.append(
                "Mild DR with hypertension increases likelihood of "
                "hypertensive vascular changes"
            )
        else:
            risk_level = "low"
            findings.append(
                "No DR detected, but hypertension can independently cause "
                "retinal vascular changes over time"
            )
    else:
        if dr_grade >= 3:
            risk_level = "low"
            findings.append(
                "Severe DR findings may overlap with hypertensive retinopathy "
                "features; blood pressure evaluation recommended"
            )

    # CDR may correlate with vascular damage
    if glaucoma_cdr is not None and glaucoma_cdr > 0.6 and has_hypertension:
        if risk_level != "high":
            risk_level = "moderate"
        findings.append(
            f"Elevated cup-to-disc ratio ({glaucoma_cdr:.2f}) with hypertension "
            "may indicate vascular optic neuropathy"
        )

    recommendation = ""
    if risk_level == "high":
        recommendation = (
            "Blood pressure assessment and dedicated hypertensive retinopathy "
            "grading by ophthalmologist recommended. Treatment of hypertension "
            "is critical to prevent further vascular damage."
        )
    elif risk_level == "moderate":
        recommendation = (
            "Monitor blood pressure regularly. Consider ophthalmology evaluation "
            "for hypertensive retinal changes at next visit."
        )
    elif risk_level == "low":
        recommendation = (
            "Low concern currently. Ensure blood pressure is monitored at "
            "routine check-ups."
        )
    else:
        recommendation = "No hypertensive retinopathy concern identified."

    return {
        "condition_name": "Hypertensive Retinopathy",
        "risk_level": _clamp_risk(risk_level),
        "findings": findings,
        "description": (
            "Hypertensive retinopathy is retinal damage caused by high blood "
            "pressure. It shares features with diabetic retinopathy including "
            "hemorrhages, exudates, and vessel abnormalities."
        ),
        "recommendation": recommendation,
    }


def _screen_amd(
    patient_age: Optional[int] = None,
    iqa_score: Optional[float] = None,
    dr_grade: int = 0,
) -> dict[str, Any]:
    """
    Screen for age-related macular degeneration (AMD) risk.

    Heuristic: AMD risk increases significantly after age 60.
    Drusen-like features (bright macular deposits) may appear in fundus
    images and partially correlate with image quality artifacts.
    """
    risk_level = "none"
    findings: list[str] = []

    if patient_age is not None:
        if patient_age >= 75:
            risk_level = "high"
            findings.append(
                f"Patient age ({patient_age}) is a strong risk factor for AMD. "
                "Prevalence increases significantly after age 75."
            )
        elif patient_age >= 60:
            risk_level = "moderate"
            findings.append(
                f"Patient age ({patient_age}) places them in the AMD risk group. "
                "Screening for drusen and macular changes is recommended."
            )
        elif patient_age >= 50:
            risk_level = "low"
            findings.append(
                f"Patient age ({patient_age}) indicates emerging AMD risk. "
                "Baseline macular assessment is advisable."
            )
    else:
        findings.append(
            "Patient age not provided. AMD risk cannot be fully assessed "
            "without age information."
        )

    # If DR grade is high, macular changes may mask or coexist with AMD
    if dr_grade >= 2 and patient_age is not None and patient_age >= 55:
        if risk_level != "high":
            risk_level = "moderate"
        findings.append(
            "DR findings may coexist with or mask early AMD changes. "
            "Dedicated macular evaluation (OCT) is recommended."
        )

    recommendation = ""
    if risk_level == "high":
        recommendation = (
            "Refer for comprehensive macular evaluation including OCT. "
            "AMD at this age group requires active monitoring and may "
            "benefit from AREDS2 supplementation."
        )
    elif risk_level == "moderate":
        recommendation = (
            "Schedule macular evaluation with OCT at the next ophthalmology visit. "
            "Monitor for symptoms like distorted vision (metamorphopsia)."
        )
    elif risk_level == "low":
        recommendation = (
            "Low AMD risk. Include macular assessment in routine eye exams. "
            "Encourage healthy diet rich in leafy greens and omega-3 fatty acids."
        )
    else:
        recommendation = "No significant AMD risk factors identified."

    return {
        "condition_name": "Age-related Macular Degeneration (AMD)",
        "risk_level": _clamp_risk(risk_level),
        "findings": findings,
        "description": (
            "AMD is a leading cause of vision loss in older adults. It affects "
            "the macula (central vision area) and can cause progressive central "
            "vision loss. Early detection allows for protective interventions."
        ),
        "recommendation": recommendation,
    }


def _screen_cataract(
    iqa_score: Optional[float] = None,
    patient_age: Optional[int] = None,
) -> dict[str, Any]:
    """
    Screen for cataract indicators based on image quality.

    Heuristic: Cataracts cause overall image haziness and reduced contrast
    in fundus photography. Low IQA scores in older patients may indicate
    media opacity from cataract.
    """
    risk_level = "none"
    findings: list[str] = []

    if iqa_score is not None:
        if iqa_score < 0.3:
            risk_level = "high"
            findings.append(
                f"Very low image quality score ({iqa_score:.2f}) suggests significant "
                "media opacity, possibly from cataract or vitreous haze."
            )
        elif iqa_score < 0.5:
            risk_level = "moderate"
            findings.append(
                f"Reduced image quality score ({iqa_score:.2f}) may indicate "
                "early lens opacity (cataract) affecting fundus visualization."
            )
        elif iqa_score < 0.7:
            risk_level = "low"
            findings.append(
                f"Mildly reduced image quality ({iqa_score:.2f}) could indicate "
                "early cataract or other media opacity."
            )

    # Age adjusts likelihood
    if patient_age is not None:
        if patient_age >= 65 and risk_level in ("moderate", "high"):
            findings.append(
                f"Patient age ({patient_age}) increases likelihood that image "
                "haziness is cataract-related."
            )
        elif patient_age >= 65 and risk_level == "none":
            risk_level = "low"
            findings.append(
                f"Patient age ({patient_age}) is a risk factor for cataract, "
                "even though image quality appears adequate."
            )

    recommendation = ""
    if risk_level == "high":
        recommendation = (
            "Significant media opacity detected. Slit-lamp examination for "
            "cataract assessment is recommended. Cataract may also be affecting "
            "the reliability of the DR screening result."
        )
    elif risk_level == "moderate":
        recommendation = (
            "Possible early cataract. Schedule a slit-lamp examination at next "
            "eye visit. If cataract is confirmed, it may need to be addressed "
            "before future retinal screening for accurate results."
        )
    elif risk_level == "low":
        recommendation = (
            "Low cataract concern. Monitor for symptoms such as blurry vision, "
            "glare sensitivity, or difficulty with night driving."
        )
    else:
        recommendation = "No cataract indicators identified from the image."

    return {
        "condition_name": "Cataract Indicators",
        "risk_level": _clamp_risk(risk_level),
        "findings": findings,
        "description": (
            "Cataracts cause clouding of the eye's natural lens, leading to "
            "blurry vision. In fundus imaging, cataracts appear as overall "
            "image haziness and reduced contrast. Diabetic patients have a "
            "higher risk of early cataract development."
        ),
        "recommendation": recommendation,
    }


def _screen_macular_edema(
    dr_grade: int,
    confidence: float,
    has_gradcam: bool = False,
) -> dict[str, Any]:
    """
    Screen for diabetic macular edema (DME) risk.

    Heuristic: DME risk correlates strongly with DR severity. Hard exudates
    near the macula (indicated by GradCAM activation patterns) increase
    suspicion. DME can occur at any DR stage but is most common at grade >= 2.
    """
    risk_level = "none"
    findings: list[str] = []

    if dr_grade >= 3:
        risk_level = "high"
        findings.append(
            "Severe or proliferative DR is strongly associated with DME. "
            "Up to 30% of patients with severe NPDR develop clinically "
            "significant macular edema."
        )
    elif dr_grade == 2:
        risk_level = "moderate"
        findings.append(
            "Moderate NPDR carries meaningful DME risk. Hard exudates "
            "and retinal thickening are common at this stage."
        )
    elif dr_grade == 1:
        risk_level = "low"
        findings.append(
            "Mild NPDR has low but non-zero DME risk. Any visual symptoms "
            "like blurring should prompt immediate OCT evaluation."
        )

    # GradCAM patterns may indicate macular involvement
    if has_gradcam and dr_grade >= 2:
        findings.append(
            "GradCAM activation patterns were generated, which may help "
            "identify areas of concern near the macula. Clinical correlation "
            "with OCT is recommended."
        )

    # High confidence in high-grade DR strengthens DME concern
    if dr_grade >= 2 and confidence > 0.8:
        findings.append(
            f"High model confidence ({confidence:.0%}) in the DR grade strengthens "
            "the DME risk assessment."
        )

    recommendation = ""
    if risk_level == "high":
        recommendation = (
            "OCT (Optical Coherence Tomography) scan of the macula is strongly "
            "recommended to evaluate for DME. If DME is confirmed, anti-VEGF "
            "therapy or laser treatment may be needed."
        )
    elif risk_level == "moderate":
        recommendation = (
            "Schedule an OCT scan to evaluate macular thickness. Report any "
            "central vision changes (blurring, distortion) to your doctor "
            "immediately."
        )
    elif risk_level == "low":
        recommendation = (
            "Low DME risk. Monitor for symptoms such as central blurring or "
            "difficulty reading. OCT at routine follow-up is advisable."
        )
    else:
        recommendation = "No significant DME risk factors identified."

    return {
        "condition_name": "Diabetic Macular Edema (DME)",
        "risk_level": _clamp_risk(risk_level),
        "findings": findings,
        "description": (
            "DME is swelling in the macula caused by fluid leakage from damaged "
            "retinal blood vessels. It is the most common cause of vision loss "
            "in diabetic patients and can occur at any stage of DR."
        ),
        "recommendation": recommendation,
    }


def screen_multi_condition(
    dr_grade: int,
    confidence: float,
    glaucoma_cdr: Optional[float] = None,
    iqa_score: Optional[float] = None,
    has_gradcam: bool = False,
    patient_age: Optional[int] = None,
    has_hypertension: bool = False,
) -> list[dict[str, Any]]:
    """
    Run multi-condition screening based on all available analysis results.

    Args:
        dr_grade: Predicted DR grade (0-4).
        confidence: Model confidence for the DR prediction (0-1).
        glaucoma_cdr: Cup-to-disc ratio from glaucoma model (optional).
        iqa_score: Image quality score from IQA model (optional).
        has_gradcam: Whether GradCAM was generated (indicates macular region analysis).
        patient_age: Patient age in years (optional).
        has_hypertension: Whether patient has hypertension history.

    Returns:
        List of condition screening results, each with:
            - condition_name, risk_level, findings, description, recommendation
    """
    conditions: list[dict[str, Any]] = []

    # 1. Hypertensive Retinopathy
    conditions.append(
        _screen_hypertensive_retinopathy(
            dr_grade=dr_grade,
            has_hypertension=has_hypertension,
            glaucoma_cdr=glaucoma_cdr,
        )
    )

    # 2. AMD Risk
    conditions.append(
        _screen_amd(
            patient_age=patient_age,
            iqa_score=iqa_score,
            dr_grade=dr_grade,
        )
    )

    # 3. Cataract Indicators
    conditions.append(
        _screen_cataract(
            iqa_score=iqa_score,
            patient_age=patient_age,
        )
    )

    # 4. Macular Edema Risk
    conditions.append(
        _screen_macular_edema(
            dr_grade=dr_grade,
            confidence=confidence,
            has_gradcam=has_gradcam,
        )
    )

    return conditions
