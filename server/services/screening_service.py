"""
Screening CRUD service layer.

Provides clean async database operations for screenings using SQLAlchemy ORM.
All functions accept an AsyncSession and operate through the ORM models.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.models.image import Image
from server.models.screening import Screening

logger = logging.getLogger(__name__)


async def create_screening(
    db: AsyncSession,
    patient_id: uuid.UUID,
    store_id: uuid.UUID,
    operator_id: uuid.UUID,
    notes: str | None = None,
) -> Screening:
    """Create a new screening record and return it."""
    screening = Screening(
        id=uuid.uuid4(),
        patient_id=patient_id,
        store_id=store_id,
        operator_id=operator_id,
        status="created",
        notes=notes,
    )
    db.add(screening)
    await db.flush()
    await db.refresh(screening)
    logger.info("Screening created: %s for patient %s", screening.id, patient_id)
    return screening


async def get_screening(
    db: AsyncSession,
    screening_id: uuid.UUID,
) -> Screening | None:
    """Fetch a single screening by ID, or None if not found."""
    result = await db.execute(
        select(Screening).where(Screening.id == screening_id)
    )
    return result.scalar_one_or_none()


async def list_screenings(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
    store_id: uuid.UUID | None = None,
    status: str | None = None,
) -> list[Screening]:
    """List screenings with optional filtering by store and status.

    Args:
        db: Async database session.
        skip: Number of rows to skip (offset).
        limit: Maximum number of rows to return.
        store_id: Filter by store UUID.
        status: Filter by screening status string.

    Returns:
        List of Screening ORM objects.
    """
    query = select(Screening).order_by(Screening.created_at.desc())

    if store_id is not None:
        query = query.where(Screening.store_id == store_id)
    if status is not None:
        query = query.where(Screening.status == status)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def count_screenings(
    db: AsyncSession,
    store_id: uuid.UUID | None = None,
    status: str | None = None,
) -> int:
    """Count screenings with optional filters (for pagination)."""
    query = select(func.count()).select_from(Screening)

    if store_id is not None:
        query = query.where(Screening.store_id == store_id)
    if status is not None:
        query = query.where(Screening.status == status)

    result = await db.execute(query)
    return result.scalar() or 0


async def update_screening_status(
    db: AsyncSession,
    screening_id: uuid.UUID,
    status: str,
) -> Screening:
    """Update the status of a screening.

    Raises:
        ValueError: If the screening is not found.
    """
    screening = await get_screening(db, screening_id)
    if screening is None:
        raise ValueError(f"Screening {screening_id} not found")

    screening.status = status

    # Set completed_at when transitioning to a terminal state
    if status in ("completed", "failed"):
        screening.completed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(screening)
    logger.info("Screening %s status updated to '%s'", screening_id, status)
    return screening


async def save_analysis_results(
    db: AsyncSession,
    screening_id: uuid.UUID,
    results: dict[str, Any],
) -> Screening:
    """Save AI analysis results to a screening.

    Expects a results dict with optional keys:
    - dr_grade_left, dr_grade_right
    - dr_confidence_left, dr_confidence_right
    - glaucoma_prob_left, glaucoma_prob_right
    - amd_prob_left, amd_prob_right
    - overall_risk, referral_required, referral_urgency, referral_reason
    - raw_results (full JSON blob)

    Raises:
        ValueError: If the screening is not found.
    """
    screening = await get_screening(db, screening_id)
    if screening is None:
        raise ValueError(f"Screening {screening_id} not found")

    # Set individual fields if present
    field_mapping = [
        "dr_grade_left", "dr_grade_right",
        "dr_confidence_left", "dr_confidence_right",
        "glaucoma_prob_left", "glaucoma_prob_right",
        "amd_prob_left", "amd_prob_right",
        "overall_risk", "referral_required",
        "referral_urgency", "referral_reason",
    ]

    for field in field_mapping:
        if field in results:
            setattr(screening, field, results[field])

    # Raw JSON blob
    if "raw_results" in results:
        screening.raw_results = results["raw_results"]

    # Mark as completed
    screening.status = "completed"
    screening.completed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(screening)
    logger.info(
        "Analysis results saved for screening %s. Risk: %s",
        screening_id,
        screening.overall_risk,
    )
    return screening
