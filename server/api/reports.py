"""Report endpoints: generate PDF, send via WhatsApp/SMS, download."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import settings
from server.dependencies import get_current_user, get_db
from server.models.report import Report
from server.models.screening import Screening
from server.schemas.report import (
    ReportDownloadResponse,
    ReportGenerateRequest,
    ReportResponse,
    ReportSendRequest,
    ReportSendResponse,
)
from server.services.report_gen import ReportGenerator
from server.services.storage import generate_presigned_url

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def generate_report(
    body: ReportGenerateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a PDF report for a completed screening."""
    # Load screening
    result = await db.execute(
        select(Screening).where(Screening.id == body.screening_id)
    )
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    if screening.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Report can only be generated for completed screenings",
        )

    # Check if report already exists
    existing = await db.execute(
        select(Report).where(Report.screening_id == body.screening_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="Report already exists for this screening. Use download endpoint.",
        )

    # Generate PDF
    generator = ReportGenerator()
    pdf_bytes, filename = await generator.generate(
        screening=screening,
        language=body.language,
        include_gradcam=body.include_gradcam,
        db=db,
    )

    # Upload to S3
    from server.services.storage import upload_image

    s3_key = f"reports/{screening.id}/{filename}"
    await upload_image(
        image_bytes=pdf_bytes,
        bucket=settings.S3_BUCKET_REPORTS,
        key=s3_key,
        content_type="application/pdf",
    )

    # Save report record
    report = Report(
        id=uuid.uuid4(),
        screening_id=screening.id,
        s3_key=s3_key,
        s3_bucket=settings.S3_BUCKET_REPORTS,
        filename=filename,
        format="pdf",
        language=body.language,
    )
    db.add(report)
    await db.flush()
    await db.refresh(report)

    logger.info("Report generated for screening %s", screening.id)
    return report


@router.post("/send", response_model=ReportSendResponse)
async def send_report(
    body: ReportSendRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a generated report via WhatsApp/SMS."""
    result = await db.execute(select(Report).where(Report.id == body.report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    channels_sent = []

    # Generate download URL
    download_url = await generate_presigned_url(
        bucket=report.s3_bucket,
        key=report.s3_key,
    )

    if "whatsapp" in body.channels:
        try:
            from server.services.notification import WhatsAppService

            wa = WhatsAppService()
            phone = body.recipient_phone
            if not phone:
                # Get from patient
                screening_result = await db.execute(
                    select(Screening).where(Screening.id == report.screening_id)
                )
                screening = screening_result.scalar_one_or_none()
                if screening and screening.patient:
                    phone = screening.patient.phone

            if phone:
                await wa.send_report(phone=phone, report_url=download_url, report=report)
                report.whatsapp_sent = True
                from datetime import datetime, timezone
                report.whatsapp_sent_at = datetime.now(timezone.utc)
                channels_sent.append("whatsapp")
        except Exception as e:
            logger.error("WhatsApp send failed: %s", e)

    if "sms" in body.channels:
        try:
            from server.services.notification import WhatsAppService

            wa = WhatsAppService()
            phone = body.recipient_phone
            if phone:
                await wa.send_sms(phone=phone, message=f"Your eye screening report is ready: {download_url}")
                report.sms_sent = True
                channels_sent.append("sms")
        except Exception as e:
            logger.error("SMS send failed: %s", e)

    await db.flush()

    return ReportSendResponse(
        report_id=report.id,
        channels_sent=channels_sent,
        message=f"Report sent via: {', '.join(channels_sent)}" if channels_sent else "No channels delivered",
    )


@router.get("/{report_id}/download", response_model=ReportDownloadResponse)
async def download_report(
    report_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a presigned download URL for a report."""
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    url = await generate_presigned_url(
        bucket=report.s3_bucket,
        key=report.s3_key,
    )

    return ReportDownloadResponse(
        download_url=url,
        expires_in=settings.S3_PRESIGNED_EXPIRY,
        filename=report.filename,
    )
