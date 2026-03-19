"""SQLAlchemy Patient model."""

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.models.user import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    state: Mapped[str | None] = mapped_column(String(100), nullable=True)
    pincode: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # ABDM / ABHA
    abha_id: Mapped[str | None] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    abha_address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    health_id_number: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Medical history flags
    is_diabetic: Mapped[bool | None] = mapped_column(default=None)
    diabetes_duration_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_hypertension: Mapped[bool | None] = mapped_column(default=None)
    medical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    screenings = relationship("Screening", back_populates="patient", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Patient {self.full_name} abha={self.abha_id}>"
