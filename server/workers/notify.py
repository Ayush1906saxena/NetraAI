"""Celery task for async WhatsApp/SMS notification delivery."""

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
    name="workers.notify.send_whatsapp",
    max_retries=3,
    default_retry_delay=60,
)
def send_whatsapp_task(
    self,
    phone: str,
    report_id: str,
):
    """Send a screening report via WhatsApp."""
    logger.info("Sending WhatsApp notification to %s for report %s", phone, report_id)

    try:
        result = _run_async(_send_whatsapp(phone, report_id))
        logger.info("WhatsApp sent to %s: %s", phone, result)
        return result
    except Exception as exc:
        logger.error("WhatsApp send failed to %s: %s", phone, exc)
        raise self.retry(exc=exc)


async def _send_whatsapp(phone: str, report_id: str) -> dict:
    from sqlalchemy import select

    from server.dependencies import async_session_factory
    from server.models.report import Report
    from server.services.notification import WhatsAppService
    from server.services.storage import generate_presigned_url

    async with async_session_factory() as db:
        result = await db.execute(
            select(Report).where(Report.id == uuid.UUID(report_id))
        )
        report = result.scalar_one_or_none()
        if not report:
            return {"error": f"Report {report_id} not found"}

        url = await generate_presigned_url(bucket=report.s3_bucket, key=report.s3_key)

        wa = WhatsAppService()
        send_result = await wa.send_report(phone=phone, report_url=url, report=report)

        # Update delivery status
        report.whatsapp_sent = True
        from datetime import datetime, timezone
        report.whatsapp_sent_at = datetime.now(timezone.utc)
        await db.commit()

        return {"status": "sent", "phone": phone, "api_response": send_result}


@celery_app.task(
    bind=True,
    name="workers.notify.send_sms",
    max_retries=3,
    default_retry_delay=60,
)
def send_sms_task(self, phone: str, message: str):
    """Send an SMS notification."""
    logger.info("Sending SMS to %s", phone)

    try:
        result = _run_async(_send_sms(phone, message))
        logger.info("SMS sent to %s: %s", phone, result)
        return result
    except Exception as exc:
        logger.error("SMS send failed to %s: %s", phone, exc)
        raise self.retry(exc=exc)


async def _send_sms(phone: str, message: str) -> dict:
    from server.services.notification import WhatsAppService

    wa = WhatsAppService()
    result = await wa.send_sms(phone=phone, message=message)
    return {"status": "sent", "phone": phone, "api_response": result}


@celery_app.task(
    bind=True,
    name="workers.notify.send_screening_complete",
    max_retries=2,
    default_retry_delay=30,
)
def send_screening_complete_notification(self, screening_id: str):
    """Send a notification when screening analysis is complete."""
    logger.info("Sending completion notification for screening %s", screening_id)

    try:
        result = _run_async(_notify_screening_complete(screening_id))
        return result
    except Exception as exc:
        logger.error("Completion notification failed for %s: %s", screening_id, exc)
        raise self.retry(exc=exc)


async def _notify_screening_complete(screening_id: str) -> dict:
    from sqlalchemy import select

    from server.dependencies import async_session_factory
    from server.models.screening import Screening
    from server.services.notification import WhatsAppService

    async with async_session_factory() as db:
        result = await db.execute(
            select(Screening).where(Screening.id == uuid.UUID(screening_id))
        )
        screening = result.scalar_one_or_none()
        if not screening:
            return {"error": "Screening not found"}

        if not screening.patient or not screening.patient.phone:
            return {"skipped": "No patient phone number"}

        wa = WhatsAppService()
        risk = screening.overall_risk or "unknown"
        message = (
            f"Your eye screening is complete. "
            f"Overall risk: {risk}. "
            f"{'Please visit an eye specialist as recommended.' if screening.referral_required else 'No immediate referral needed.'} "
            f"Your detailed report will be shared shortly."
        )

        send_result = await wa.send_text_message(
            phone=screening.patient.phone,
            text=message,
        )
        return {"status": "sent", "phone": screening.patient.phone, "result": send_result}
