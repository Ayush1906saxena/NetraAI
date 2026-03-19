"""Screening workflow orchestration service."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.models.image import Image
from server.models.screening import Screening
from server.services.inference import ModelRegistry
from server.services.referral import ReferralService
from server.services.storage import download_image

logger = logging.getLogger(__name__)


class ScreeningService:
    """Orchestrates the full screening workflow: create, analyze, finalize."""

    def __init__(self, db: AsyncSession, models: ModelRegistry):
        self.db = db
        self.models = models
        self.referral_service = ReferralService()

    async def create_screening(
        self,
        patient_id: uuid.UUID,
        store_id: uuid.UUID,
        operator_id: uuid.UUID,
        notes: str | None = None,
    ) -> Screening:
        """Create a new screening record."""
        screening = Screening(
            id=uuid.uuid4(),
            patient_id=patient_id,
            store_id=store_id,
            operator_id=operator_id,
            status="created",
            notes=notes,
        )
        self.db.add(screening)
        await self.db.flush()
        await self.db.refresh(screening)
        logger.info("Screening created: %s", screening.id)
        return screening

    async def run_analysis(
        self,
        screening_id: uuid.UUID,
        model_names: list[str] | None = None,
    ) -> Screening:
        """Run AI analysis on all approved images for a screening."""
        if model_names is None:
            model_names = ["dr", "glaucoma", "amd"]

        result = await self.db.execute(
            select(Screening).where(Screening.id == screening_id)
        )
        screening = result.scalar_one_or_none()
        if not screening:
            raise ValueError(f"Screening {screening_id} not found")

        # Fetch approved images
        img_result = await self.db.execute(
            select(Image).where(
                Image.screening_id == screening_id,
                Image.iqa_passed == True,
            )
        )
        images = img_result.scalars().all()
        if not images:
            raise ValueError("No quality-approved images found for analysis")

        screening.status = "analyzing"
        await self.db.flush()

        # Analyze each image
        for img in images:
            image_bytes = await download_image(bucket=img.s3_bucket, key=img.s3_key)
            analysis = await self.models.analyze_fundus(image_bytes, model_names=model_names)
            img.ai_results = analysis

            eye = img.eye
            if "dr" in analysis:
                setattr(screening, f"dr_grade_{eye}", analysis["dr"]["grade"])
                setattr(screening, f"dr_confidence_{eye}", analysis["dr"]["confidence"])
            if "glaucoma" in analysis:
                setattr(screening, f"glaucoma_prob_{eye}", analysis["glaucoma"]["probability"])
            if "amd" in analysis:
                setattr(screening, f"amd_prob_{eye}", analysis["amd"]["probability"])

        # Compute referral
        referral = self.referral_service.compute_referral(screening)
        screening.referral_required = referral["referral_required"]
        screening.referral_urgency = referral["referral_urgency"]
        screening.referral_reason = referral["referral_reason"]
        screening.overall_risk = referral["overall_risk"]

        screening.status = "completed"
        screening.completed_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self.db.refresh(screening)

        logger.info(
            "Screening %s completed. Risk: %s, Referral: %s",
            screening_id,
            screening.overall_risk,
            screening.referral_required,
        )
        return screening

    async def get_screening_with_details(self, screening_id: uuid.UUID) -> dict[str, Any]:
        """Get screening with all related data."""
        result = await self.db.execute(
            select(Screening).where(Screening.id == screening_id)
        )
        screening = result.scalar_one_or_none()
        if not screening:
            raise ValueError(f"Screening {screening_id} not found")

        img_result = await self.db.execute(
            select(Image).where(Image.screening_id == screening_id)
        )
        images = img_result.scalars().all()

        return {
            "screening": screening,
            "images": images,
            "patient": screening.patient,
            "store": screening.store,
            "report": screening.report,
        }
