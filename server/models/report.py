"""SQLAlchemy Report model."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.models.user import Base


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    screening_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("screenings.id"), unique=True, nullable=False, index=True
    )
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False, default="pdf")
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")

    # Delivery status
    whatsapp_sent: Mapped[bool] = mapped_column(default=False)
    sms_sent: Mapped[bool] = mapped_column(default=False)
    email_sent: Mapped[bool] = mapped_column(default=False)
    whatsapp_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ABDM
    abdm_pushed: Mapped[bool] = mapped_column(default=False)
    abdm_record_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    screening = relationship("Screening", back_populates="report", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Report {self.id} screening={self.screening_id}>"
