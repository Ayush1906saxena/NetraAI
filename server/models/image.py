"""SQLAlchemy Image model."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.models.user import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    screening_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("screenings.id"), nullable=False, index=True
    )
    eye: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # left, right
    image_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="fundus"
    )  # fundus, oct, anterior

    # Storage
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False, default="image/jpeg")
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Image Quality Assessment
    iqa_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    iqa_passed: Mapped[bool | None] = mapped_column(default=None)
    iqa_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Grad-CAM / explainability
    gradcam_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # AI results specific to this image
    ai_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    screening = relationship("Screening", back_populates="images", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Image {self.id} eye={self.eye} iqa={self.iqa_score}>"
