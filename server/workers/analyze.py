"""Celery task for async AI inference and report generation."""

import asyncio
import logging
import uuid

from server.workers import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="workers.analyze.run_analysis",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def run_analysis_task(self, screening_id: str, model_names: list[str] | None = None):
    """
    Run AI analysis on a screening's images.

    This is the Celery task version of the screening analysis pipeline,
    used when synchronous analysis fails or for batch processing.
    """
    logger.info("Starting async analysis for screening %s", screening_id)

    try:
        result = _run_async(_analyze(screening_id, model_names))
        logger.info("Async analysis completed for screening %s: %s", screening_id, result.get("status"))
        return result
    except Exception as exc:
        logger.error("Analysis task failed for %s: %s", screening_id, exc)
        raise self.retry(exc=exc)


async def _analyze(screening_id: str, model_names: list[str] | None) -> dict:
    """Async analysis implementation."""
    from server.dependencies import async_session_factory
    from server.services.inference import ModelRegistry
    from server.services.screening import ScreeningService
    from server.config import settings

    # Load models
    registry = ModelRegistry()
    await registry.load_models(
        dr_path=settings.MODEL_DR_PATH,
        glaucoma_path=settings.MODEL_GLAUCOMA_PATH,
        amd_path=settings.MODEL_AMD_PATH,
        iqa_path=settings.MODEL_IQA_PATH,
        segmentation_path=settings.MODEL_SEGMENTATION_PATH,
        device=settings.MODEL_DEVICE,
    )

    async with async_session_factory() as db:
        try:
            service = ScreeningService(db=db, models=registry)
            screening = await service.run_analysis(
                screening_id=uuid.UUID(screening_id),
                model_names=model_names,
            )
            await db.commit()

            return {
                "status": "completed",
                "screening_id": screening_id,
                "overall_risk": screening.overall_risk,
                "referral_required": screening.referral_required,
            }
        except Exception as e:
            await db.rollback()
            raise
        finally:
            await registry.unload_models()


@celery_app.task(
    bind=True,
    name="workers.analyze.generate_report_task",
    max_retries=2,
    default_retry_delay=15,
)
def generate_report_task(self, screening_id: str, language: str = "en"):
    """Generate a PDF report for a completed screening."""
    logger.info("Starting report generation for screening %s", screening_id)

    try:
        result = _run_async(_generate_report(screening_id, language))
        logger.info("Report generated for screening %s", screening_id)
        return result
    except Exception as exc:
        logger.error("Report generation failed for %s: %s", screening_id, exc)
        raise self.retry(exc=exc)


async def _generate_report(screening_id: str, language: str) -> dict:
    """Async report generation."""
    from sqlalchemy import select

    from server.config import settings
    from server.dependencies import async_session_factory
    from server.models.report import Report
    from server.models.screening import Screening
    from server.services.report_gen import ReportGenerator
    from server.services.storage import upload_image

    async with async_session_factory() as db:
        try:
            result = await db.execute(
                select(Screening).where(Screening.id == uuid.UUID(screening_id))
            )
            screening = result.scalar_one_or_none()
            if not screening:
                return {"error": f"Screening {screening_id} not found"}

            generator = ReportGenerator()
            pdf_bytes, filename = await generator.generate(
                screening=screening,
                language=language,
                db=db,
            )

            s3_key = f"reports/{screening_id}/{filename}"
            await upload_image(
                image_bytes=pdf_bytes,
                bucket=settings.S3_BUCKET_REPORTS,
                key=s3_key,
                content_type="application/pdf",
            )

            report = Report(
                id=uuid.uuid4(),
                screening_id=screening.id,
                s3_key=s3_key,
                s3_bucket=settings.S3_BUCKET_REPORTS,
                filename=filename,
                format="pdf",
                language=language,
            )
            db.add(report)
            await db.commit()

            return {"status": "generated", "report_id": str(report.id), "filename": filename}
        except Exception as e:
            await db.rollback()
            raise
