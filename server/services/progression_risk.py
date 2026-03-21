"""
Severity Progression Risk Estimator.

Estimates the risk of DR progression based on current grade, model confidence,
and optional clinical factors (diabetes duration, HbA1c).

Based on published DR progression rates from clinical literature:
  - Klein et al. (WESDR study)
  - UKPDS, DCCT/EDIC longitudinal studies
"""

from typing import Any, Optional


# Base annual progression rates (probability of advancing one grade per year)
BASE_PROGRESSION_RATES: dict[int, float] = {
    0: 0.05,   # Grade 0 -> 1: ~5% per year
    1: 0.12,   # Grade 1 -> 2: ~12% per year
    2: 0.25,   # Grade 2 -> 3: ~25% per year
    3: 0.52,   # Grade 3 -> 4: ~52% per year
    4: 0.0,    # Grade 4: already at highest grade
}

# Risk level thresholds (based on 1-year progression probability)
RISK_THRESHOLDS = {
    "low": 0.10,
    "moderate": 0.20,
    "high": 0.40,
}

# Recommended rescreening intervals (months)
RESCREEN_MAP = {
    "low": 12,
    "moderate": 6,
    "high": 3,
    "very_high": 1,
}

RISK_EXPLANATIONS = {
    "low": (
        "The estimated progression risk is low. Current DR stage and risk factors "
        "suggest stable disease. Maintain routine screening schedule and good "
        "glycemic control."
    ),
    "moderate": (
        "The estimated progression risk is moderate. Closer monitoring is advised. "
        "Focus on optimizing blood sugar and blood pressure control to slow progression."
    ),
    "high": (
        "The estimated progression risk is high. Active management and frequent "
        "monitoring are critical. Consult with an ophthalmologist about preventive "
        "treatments and tighten metabolic control."
    ),
    "very_high": (
        "The estimated progression risk is very high. Urgent specialist involvement "
        "is needed. The combination of advanced DR stage and risk factors suggests "
        "rapid progression is likely without intervention."
    ),
}


def estimate_progression_risk(
    current_grade: int,
    confidence: float,
    diabetes_duration_years: Optional[float] = None,
    hba1c: Optional[float] = None,
) -> dict[str, Any]:
    """
    Estimate DR progression risk based on current findings and clinical factors.

    Args:
        current_grade: DR grade (0-4).
        confidence: Model confidence for the predicted grade (0-1).
        diabetes_duration_years: Years since diabetes diagnosis (optional).
        hba1c: Most recent HbA1c value in % (optional).

    Returns:
        dict with:
            - progression_risk_1yr (float, 0-1)
            - progression_risk_5yr (float, 0-1)
            - risk_level ("low" / "moderate" / "high" / "very_high")
            - risk_factors (list[str])
            - recommended_rescreen_months (int)
            - explanation (str)
    """
    if current_grade < 0 or current_grade > 4:
        raise ValueError(f"Invalid DR grade: {current_grade}. Must be 0-4.")

    base_rate = BASE_PROGRESSION_RATES[current_grade]
    risk_factors: list[str] = []
    multiplier = 1.0

    # --- Adjustment: model confidence ---
    # Lower confidence means more uncertainty -> widen risk estimate upward
    if confidence < 0.5:
        multiplier *= 1.3
        risk_factors.append(
            f"Low model confidence ({confidence:.0%}) increases uncertainty in risk estimate"
        )
    elif confidence < 0.7:
        multiplier *= 1.1
        risk_factors.append(
            f"Moderate model confidence ({confidence:.0%}) slightly increases uncertainty"
        )

    # --- Adjustment: diabetes duration ---
    if diabetes_duration_years is not None:
        if diabetes_duration_years > 20:
            multiplier *= 1.8
            risk_factors.append(
                f"Long diabetes duration ({diabetes_duration_years:.0f} years, >20yr) "
                "significantly increases progression risk"
            )
        elif diabetes_duration_years > 10:
            multiplier *= 1.5
            risk_factors.append(
                f"Extended diabetes duration ({diabetes_duration_years:.0f} years, >10yr) "
                "increases progression risk by ~50%"
            )
        elif diabetes_duration_years > 5:
            multiplier *= 1.2
            risk_factors.append(
                f"Diabetes duration of {diabetes_duration_years:.0f} years moderately "
                "increases progression risk"
            )

    # --- Adjustment: HbA1c (glycemic control) ---
    if hba1c is not None:
        if hba1c > 10:
            multiplier *= 1.6
            risk_factors.append(
                f"Very poor glycemic control (HbA1c {hba1c:.1f}%, >10%) "
                "substantially increases progression risk"
            )
        elif hba1c > 8:
            multiplier *= 1.3
            risk_factors.append(
                f"Poor glycemic control (HbA1c {hba1c:.1f}%, >8%) "
                "increases progression risk by ~30%"
            )
        elif hba1c > 7:
            multiplier *= 1.1
            risk_factors.append(
                f"Suboptimal glycemic control (HbA1c {hba1c:.1f}%, >7%) "
                "slightly increases progression risk"
            )
        else:
            risk_factors.append(
                f"Good glycemic control (HbA1c {hba1c:.1f}%) is protective against progression"
            )

    # Grade-based risk factor
    grade_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    if current_grade >= 3:
        risk_factors.insert(0, f"Current grade ({grade_names[current_grade]}) carries inherently high progression risk")
    elif current_grade == 2:
        risk_factors.insert(0, f"Current grade ({grade_names[current_grade]}) has moderate baseline progression risk")
    elif current_grade == 0:
        if not risk_factors:
            risk_factors.append("No DR detected; baseline progression risk is low")

    # Compute adjusted 1-year risk
    risk_1yr = min(base_rate * multiplier, 0.95)

    # Compute 5-year cumulative risk: 1 - (1 - annual_risk)^5
    risk_5yr = 1.0 - ((1.0 - risk_1yr) ** 5)
    risk_5yr = min(risk_5yr, 0.99)

    # Grade 4 special case: already at max, risk is about complications, not progression
    if current_grade == 4:
        risk_1yr = 0.0
        risk_5yr = 0.0
        risk_factors = [
            "Already at the most advanced stage (Proliferative DR). "
            "Risk is now about complications (vitreous hemorrhage, retinal detachment) "
            "rather than grade progression.",
            "Active treatment (anti-VEGF, laser, vitrectomy) is critical to prevent vision loss.",
        ]

    # Determine risk level
    if current_grade == 4:
        risk_level = "very_high"
    elif risk_1yr >= RISK_THRESHOLDS["high"]:
        risk_level = "very_high"
    elif risk_1yr >= RISK_THRESHOLDS["moderate"]:
        risk_level = "high"
    elif risk_1yr >= RISK_THRESHOLDS["low"]:
        risk_level = "moderate"
    else:
        risk_level = "low"

    rescreen_months = RESCREEN_MAP[risk_level]
    explanation = RISK_EXPLANATIONS[risk_level]

    return {
        "progression_risk_1yr": round(risk_1yr, 4),
        "progression_risk_5yr": round(risk_5yr, 4),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommended_rescreen_months": rescreen_months,
        "explanation": explanation,
    }
