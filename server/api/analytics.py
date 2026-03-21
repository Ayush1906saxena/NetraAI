"""
Population Analytics API — aggregate statistics for screenings.

Returns mock data structured so it can easily switch to real DB queries later.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Path

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Seed random for consistent mock data within a session ─────────────────
_rng = random.Random(42)


def _mock_grade_distribution() -> dict[str, int]:
    """Realistic DR grade distribution: most screenings are grade 0-1."""
    return {
        "No DR": 4215,
        "Mild NPDR": 1038,
        "Moderate NPDR": 567,
        "Severe NPDR": 189,
        "Proliferative DR": 73,
    }


def _mock_summary() -> dict[str, Any]:
    """Generate realistic summary statistics."""
    grade_dist = _mock_grade_distribution()
    total = sum(grade_dist.values())
    referable = grade_dist["Moderate NPDR"] + grade_dist["Severe NPDR"] + grade_dist["Proliferative DR"]

    return {
        "total_screenings": total,
        "total_patients": int(total * 0.87),  # some patients screened more than once
        "total_stores": 24,
        "referable_rate": round(referable / total * 100, 1),
        "grade_distribution": grade_dist,
        "average_confidence": 0.847,
        "iqa_rejection_rate": 8.3,
        "screenings_today": 47,
        "screenings_this_week": 312,
        "screenings_this_month": 1284,
        "period": {
            "start": (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z",
            "end": datetime.utcnow().isoformat() + "Z",
        },
    }


def _mock_demographics() -> dict[str, Any]:
    """Generate realistic demographic breakdowns."""
    return {
        "age_distribution": {
            "<30": 312,
            "30-40": 876,
            "40-50": 1843,
            "50-60": 1956,
            "60+": 1095,
        },
        "gender_distribution": {
            "male": 3247,
            "female": 2812,
            "other": 23,
        },
        "diabetes_duration_vs_grade": {
            "description": "Average diabetes duration (years) per DR grade",
            "data": [
                {"grade": 0, "grade_name": "No DR", "avg_duration_years": 4.2, "count": 4215},
                {"grade": 1, "grade_name": "Mild NPDR", "avg_duration_years": 7.8, "count": 1038},
                {"grade": 2, "grade_name": "Moderate NPDR", "avg_duration_years": 11.3, "count": 567},
                {"grade": 3, "grade_name": "Severe NPDR", "avg_duration_years": 14.6, "count": 189},
                {"grade": 4, "grade_name": "Proliferative DR", "avg_duration_years": 18.1, "count": 73},
            ],
        },
        "diabetes_type_distribution": {
            "type_1": 487,
            "type_2": 5432,
            "gestational": 78,
            "unknown": 85,
        },
    }


def _mock_store_metrics(store_id: str) -> dict[str, Any]:
    """Generate per-store metrics from a deterministic seed based on store_id."""
    # Use store_id as seed for reproducible per-store data
    rng = random.Random(store_id)

    screening_count = rng.randint(120, 600)
    referral_count = rng.randint(int(screening_count * 0.08), int(screening_count * 0.2))

    # Generate a realistic grade distribution for this store
    grades = {
        "No DR": int(screening_count * rng.uniform(0.55, 0.75)),
        "Mild NPDR": int(screening_count * rng.uniform(0.12, 0.22)),
        "Moderate NPDR": int(screening_count * rng.uniform(0.06, 0.12)),
        "Severe NPDR": int(screening_count * rng.uniform(0.02, 0.06)),
        "Proliferative DR": int(screening_count * rng.uniform(0.005, 0.02)),
    }
    # Adjust No DR to match total
    grades["No DR"] = screening_count - sum(v for k, v in grades.items() if k != "No DR")

    # Top grade is the highest grade with nonzero count
    top_grade = 0
    grade_names = list(grades.keys())
    for i in range(len(grade_names) - 1, -1, -1):
        if grades[grade_names[i]] > 0:
            top_grade = i
            break

    # Monthly trend (last 6 months)
    now = datetime.utcnow()
    monthly_trend = []
    for i in range(5, -1, -1):
        month_date = now - timedelta(days=30 * i)
        monthly_trend.append({
            "month": month_date.strftime("%Y-%m"),
            "screenings": rng.randint(15, 90),
            "referrals": rng.randint(2, 15),
        })

    return {
        "store_id": store_id,
        "screening_count": screening_count,
        "referral_rate": round(referral_count / screening_count * 100, 1),
        "avg_confidence": round(rng.uniform(0.78, 0.93), 3),
        "top_grade": top_grade,
        "top_grade_name": grade_names[top_grade],
        "grade_distribution": grades,
        "iqa_rejection_rate": round(rng.uniform(3.0, 15.0), 1),
        "monthly_trend": monthly_trend,
        "active_since": (now - timedelta(days=rng.randint(90, 365))).isoformat() + "Z",
    }


@router.get("/summary")
async def analytics_summary() -> dict[str, Any]:
    """
    Aggregate screening statistics across all stores.

    Returns total_screenings, referable_rate, grade_distribution,
    average_confidence, iqa_rejection_rate, and time-period counts.
    """
    return {
        "status": "success",
        "data": _mock_summary(),
    }


@router.get("/demographics")
async def analytics_demographics() -> dict[str, Any]:
    """
    Demographic breakdowns of screened patients.

    Returns age_distribution, gender_distribution, and
    diabetes_duration_vs_grade correlation data.
    """
    return {
        "status": "success",
        "data": _mock_demographics(),
    }


@router.get("/store/{store_id}")
async def analytics_store(
    store_id: str = Path(..., description="Unique store identifier"),
) -> dict[str, Any]:
    """
    Per-store screening metrics.

    Returns screening_count, referral_rate, avg_confidence, top_grade,
    grade_distribution, and monthly_trend.
    """
    return {
        "status": "success",
        "data": _mock_store_metrics(store_id),
    }
