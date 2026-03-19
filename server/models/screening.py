"""SQLAlchemy Screening model."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.models.user import Base


class Screening(Base):
    __tablename__ = "screenings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True
    )
    store_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id"), nullable=False, index=True
    )
    operator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="created"
    )  # created, images_uploaded, analyzing, completed, failed, referred

    # AI Results
    dr_grade_left: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dr_grade_right: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dr_confidence_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    dr_confidence_right: Mapped[float | None] = mapped_column(Float, nullable=True)

    glaucoma_prob_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    glaucoma_prob_right: Mapped[float | None] = mapped_column(Float, nullable=True)

    amd_prob_left: Mapped[float | None] = mapped_column(Float, nullable=True)
    amd_prob_right: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Composite risk
    overall_risk: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # low, moderate, high, urgent
    referral_required: Mapped[bool | None] = mapped_column(default=None)
    referral_urgency: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # routine, urgent, emergency
    referral_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full JSON results blob for extensibility
    raw_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    screened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    patient = relationship("Patient", back_populates="screenings", lazy="selectin")
    store = relationship("Store", back_populates="screenings", lazy="selectin")
    operator = relationship("User", back_populates="screenings", lazy="selectin")
    images = relationship("Image", back_populates="screening", lazy="selectin")
    report = relationship("Report", back_populates="screening", uselist=False, lazy="selectin")

    def __repr__(self) -> str:
        return f"<Screening {self.id} status={self.status}>"
