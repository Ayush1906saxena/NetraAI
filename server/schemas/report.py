"""Pydantic schemas for report generation and delivery."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ReportGenerateRequest(BaseModel):
    screening_id: UUID
    language: str = Field("en", pattern=r"^(en|hi|ta|te|kn|mr)$")
    include_gradcam: bool = True


class ReportSendRequest(BaseModel):
    report_id: UUID
    channels: list[str] = Field(
        default=["whatsapp"],
        description="Delivery channels: whatsapp, sms, email",
    )
    recipient_phone: str | None = None
    recipient_email: str | None = None


class ReportResponse(BaseModel):
    id: UUID
    screening_id: UUID
    s3_key: str
    filename: str
    format: str
    language: str
    whatsapp_sent: bool
    sms_sent: bool
    email_sent: bool
    abdm_pushed: bool
    generated_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportSendResponse(BaseModel):
    report_id: UUID
    channels_sent: list[str]
    message: str


class ReportDownloadResponse(BaseModel):
    download_url: str
    expires_in: int
    filename: str
