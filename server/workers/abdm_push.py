"""Celery task for async ABDM health record push."""

import asyncio
import logging
import uuid

from server.workers import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="workers.abdm_push.push_health_record",
    max_retries=3,
    default_retry_delay=120,
)
def push_health_record_task(self, screening_id: str):
    """Push screening results to ABDM as a FHIR health record."""
    logger.info("Starting ABDM push for screening %s", screening_id)

    try:
        result = _run_async(_push_record(screening_id))
        logger.info("ABDM push result for %s: %s", screening_id, result)
        return result
    except Exception as exc:
        logger.error("ABDM push failed for %s: %s", screening_id, exc)
        raise self.retry(exc=exc)


async def _push_record(screening_id: str) -> dict:
    from sqlalchemy import select

    from server.dependencies import async_session_factory
    from server.models.report import Report
    from server.models.screening import Screening
    from server.services.abdm import ABDMClient

    async with async_session_factory() as db:
        result = await db.execute(
            select(Screening).where(Screening.id == uuid.UUID(screening_id))
        )
        screening = result.scalar_one_or_none()
        if not screening:
            return {"error": f"Screening {screening_id} not found"}

        if not screening.patient or not screening.patient.abha_id:
            return {"skipped": "Patient has no ABHA ID"}

        # Build screening data dict
        screening_data = {
            "screening_id": str(screening.id),
            "dr_grade_left": screening.dr_grade_left,
            "dr_grade_right": screening.dr_grade_right,
            "dr_confidence_left": screening.dr_confidence_left,
            "dr_confidence_right": screening.dr_confidence_right,
            "glaucoma_prob_left": screening.glaucoma_prob_left,
            "glaucoma_prob_right": screening.glaucoma_prob_right,
            "amd_prob_left": screening.amd_prob_left,
            "amd_prob_right": screening.amd_prob_right,
            "overall_risk": screening.overall_risk,
            "referral_required": screening.referral_required,
            "referral_urgency": screening.referral_urgency,
            "referral_reason": screening.referral_reason,
        }

        abdm = ABDMClient()

        # Link care context first
        link_result = await abdm.link_care_context(
            patient_abha_id=screening.patient.abha_id,
            screening_id=str(screening.id),
            display_name=f"Eye Screening - {screening.screened_at.strftime('%d %b %Y') if screening.screened_at else 'N/A'}",
        )

        # Push health record
        push_result = await abdm.push_health_record(
            patient_abha_id=screening.patient.abha_id,
            screening_data=screening_data,
            care_context_reference=str(screening.id),
        )

        # Update report ABDM status if report exists
        report_result = await db.execute(
            select(Report).where(Report.screening_id == screening.id)
        )
        report = report_result.scalar_one_or_none()
        if report and push_result.get("pushed"):
            report.abdm_pushed = True
            report.abdm_record_id = push_result.get("data", {}).get("requestId", str(uuid.uuid4()))

        await db.commit()

        return {
            "status": "pushed" if push_result.get("pushed") else "failed",
            "screening_id": screening_id,
            "abha_id": screening.patient.abha_id,
            "link_result": link_result,
            "push_result": push_result,
        }


@celery_app.task(
    bind=True,
    name="workers.abdm_push.verify_abha",
    max_retries=2,
    default_retry_delay=30,
)
def verify_abha_task(self, abha_id: str) -> dict:
    """Verify an ABHA ID with the ABDM gateway."""
    logger.info("Verifying ABHA ID: %s", abha_id)

    try:
        result = _run_async(_verify(abha_id))
        return result
    except Exception as exc:
        logger.error("ABHA verification failed for %s: %s", abha_id, exc)
        raise self.retry(exc=exc)


async def _verify(abha_id: str) -> dict:
    from server.services.abdm import ABDMClient

    client = ABDMClient()
    return await client.verify_abha(abha_id)
