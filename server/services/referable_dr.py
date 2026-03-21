"""
Binary Referable DR Classifier.

Takes 5-class DR probabilities and computes a binary referable/non-referable
decision. Referable DR is defined as grade >= 2 (moderate NPDR or worse).

Based on the International Clinical Diabetic Retinopathy (ICDR) scale:
  Grade 0: No DR
  Grade 1: Mild NPDR
  Grade 2: Moderate NPDR  (referable)
  Grade 3: Severe NPDR    (referable)
  Grade 4: Proliferative  (referable)
"""

from typing import Any


# Thresholds
REFERABLE_THRESHOLD = 0.5
HIGH_CONFIDENCE_UPPER = 0.8
HIGH_CONFIDENCE_LOWER = 0.2

CLINICAL_ACTIONS = {
    "referable_high": (
        "Refer to ophthalmologist. The AI model is highly confident that this "
        "patient has referable diabetic retinopathy (grade 2 or higher). "
        "Prompt specialist evaluation is recommended."
    ),
    "referable_medium": (
        "Refer to ophthalmologist. The model indicates likely referable diabetic "
        "retinopathy, but confidence is moderate. A comprehensive dilated eye "
        "exam is recommended to confirm."
    ),
    "referable_low": (
        "The model suggests possible referable diabetic retinopathy, but "
        "confidence is low. Consider re-screening with a higher quality image "
        "or schedule an in-person exam for confirmation."
    ),
    "non_referable_high": (
        "No referral needed at this time. The AI model is highly confident "
        "that this patient does not have referable diabetic retinopathy. "
        "Continue routine annual screening."
    ),
    "non_referable_medium": (
        "No immediate referral indicated, but confidence is moderate. "
        "Consider re-screening in 6 months or sooner if symptoms develop."
    ),
    "non_referable_low": (
        "The model does not detect referable DR, but confidence is low. "
        "This may be due to image quality issues. Re-screening with a "
        "better quality image is recommended."
    ),
}


def classify_referable_dr(probabilities_list: list[float]) -> dict[str, Any]:
    """
    Classify whether DR is referable based on 5-class probabilities.

    Args:
        probabilities_list: List of 5 floats [P(grade0), P(grade1), ..., P(grade4)].

    Returns:
        dict with:
            - is_referable (bool)
            - referable_probability (float)
            - confidence_level ("high" / "medium" / "low")
            - clinical_action (str)
            - explanation (str)
    """
    if len(probabilities_list) != 5:
        raise ValueError(
            f"Expected 5 class probabilities, got {len(probabilities_list)}"
        )

    # P(referable) = P(grade >= 2) = sum of probs for grades 2, 3, 4
    referable_prob = float(sum(probabilities_list[2:5]))
    # Clamp to [0, 1]
    referable_prob = max(0.0, min(1.0, referable_prob))

    is_referable = referable_prob >= REFERABLE_THRESHOLD

    # Determine confidence level
    if referable_prob > HIGH_CONFIDENCE_UPPER or referable_prob < HIGH_CONFIDENCE_LOWER:
        confidence_level = "high"
    elif referable_prob > 0.65 or referable_prob < 0.35:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # Select clinical action
    if is_referable:
        action_key = f"referable_{confidence_level}"
    else:
        action_key = f"non_referable_{confidence_level}"
    clinical_action = CLINICAL_ACTIONS[action_key]

    # Build explanation
    non_referable_prob = float(sum(probabilities_list[0:2]))
    explanation = (
        f"The probability of referable DR (grade >= 2) is {referable_prob:.1%}. "
        f"The probability of non-referable findings (grade 0-1) is {non_referable_prob:.1%}. "
        f"Using a threshold of {REFERABLE_THRESHOLD:.0%}, this is classified as "
        f"{'referable' if is_referable else 'non-referable'} with {confidence_level} confidence."
    )

    return {
        "is_referable": is_referable,
        "referable_probability": round(referable_prob, 4),
        "confidence_level": confidence_level,
        "clinical_action": clinical_action,
        "explanation": explanation,
    }
