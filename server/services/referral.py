"""Referral logic service — determines if and how urgently a patient should be referred."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds
DR_MILD_THRESHOLD = 1
DR_MODERATE_THRESHOLD = 2
DR_SEVERE_THRESHOLD = 3
GLAUCOMA_SUSPECT_THRESHOLD = 0.5
GLAUCOMA_HIGH_THRESHOLD = 0.75
AMD_SUSPECT_THRESHOLD = 0.5
AMD_HIGH_THRESHOLD = 0.75


class ReferralService:
    """Compute referral recommendations from screening AI results."""

    def compute_referral(self, screening: Any) -> dict:
        """
        Analyze screening results and return referral recommendation.

        Returns:
            dict with keys: referral_required, referral_urgency, referral_reason, overall_risk
        """
        reasons: list[str] = []
        urgency_level = 0  # 0=none, 1=routine, 2=urgent, 3=emergency

        # ── Diabetic Retinopathy ─────────────────────────────────────
        for eye in ("left", "right"):
            grade = getattr(screening, f"dr_grade_{eye}", None)
            confidence = getattr(screening, f"dr_confidence_{eye}", None)
            if grade is None:
                continue

            conf_str = f" (confidence: {confidence:.0%})" if confidence else ""
            if grade >= 4:
                reasons.append(f"Proliferative DR in {eye} eye{conf_str}")
                urgency_level = max(urgency_level, 3)
            elif grade >= DR_SEVERE_THRESHOLD:
                reasons.append(f"Severe NPDR in {eye} eye{conf_str}")
                urgency_level = max(urgency_level, 3)
            elif grade >= DR_MODERATE_THRESHOLD:
                reasons.append(f"Moderate NPDR in {eye} eye{conf_str}")
                urgency_level = max(urgency_level, 2)
            elif grade >= DR_MILD_THRESHOLD:
                reasons.append(f"Mild NPDR in {eye} eye{conf_str}")
                urgency_level = max(urgency_level, 1)

        # ── Glaucoma ─────────────────────────────────────────────────
        for eye in ("left", "right"):
            prob = getattr(screening, f"glaucoma_prob_{eye}", None)
            if prob is None:
                continue

            if prob >= GLAUCOMA_HIGH_THRESHOLD:
                reasons.append(f"High glaucoma risk in {eye} eye ({prob:.0%})")
                urgency_level = max(urgency_level, 2)
            elif prob >= GLAUCOMA_SUSPECT_THRESHOLD:
                reasons.append(f"Glaucoma suspect in {eye} eye ({prob:.0%})")
                urgency_level = max(urgency_level, 1)

        # ── AMD ──────────────────────────────────────────────────────
        for eye in ("left", "right"):
            prob = getattr(screening, f"amd_prob_{eye}", None)
            if prob is None:
                continue

            if prob >= AMD_HIGH_THRESHOLD:
                reasons.append(f"High AMD risk in {eye} eye ({prob:.0%})")
                urgency_level = max(urgency_level, 2)
            elif prob >= AMD_SUSPECT_THRESHOLD:
                reasons.append(f"AMD suspect in {eye} eye ({prob:.0%})")
                urgency_level = max(urgency_level, 1)

        # ── Map urgency level ────────────────────────────────────────
        urgency_map = {0: None, 1: "routine", 2: "urgent", 3: "emergency"}
        risk_map = {0: "low", 1: "moderate", 2: "high", 3: "urgent"}

        referral_required = urgency_level > 0

        return {
            "referral_required": referral_required,
            "referral_urgency": urgency_map[urgency_level],
            "referral_reason": "; ".join(reasons) if reasons else None,
            "overall_risk": risk_map[urgency_level],
        }

    def get_specialist_type(self, screening: Any) -> list[str]:
        """Determine which specialist(s) to refer to."""
        specialists = set()

        for eye in ("left", "right"):
            grade = getattr(screening, f"dr_grade_{eye}", None)
            if grade and grade >= DR_MODERATE_THRESHOLD:
                specialists.add("retina_specialist")

            glaucoma = getattr(screening, f"glaucoma_prob_{eye}", None)
            if glaucoma and glaucoma >= GLAUCOMA_SUSPECT_THRESHOLD:
                specialists.add("glaucoma_specialist")

            amd = getattr(screening, f"amd_prob_{eye}", None)
            if amd and amd >= AMD_SUSPECT_THRESHOLD:
                specialists.add("retina_specialist")

        if not specialists:
            specialists.add("general_ophthalmologist")

        return sorted(specialists)
