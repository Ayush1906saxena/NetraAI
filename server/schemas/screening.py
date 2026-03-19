"""Pydantic schemas for screening CRUD."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ScreeningCreate(BaseModel):
    patient_id: UUID
    store_id: UUID
    notes: str | None = None


class ScreeningUpdate(BaseModel):
    notes: str | None = None
    status: str | None = None


class ImageResult(BaseModel):
    eye: str
    iqa_score: float | None = None
    iqa_passed: bool | None = None
    dr_grade: int | None = None
    dr_confidence: float | None = None
    glaucoma_prob: float | None = None
    amd_prob: float | None = None


class ScreeningResponse(BaseModel):
    id: UUID
    patient_id: UUID
    store_id: UUID
    operator_id: UUID
    status: str
    dr_grade_left: int | None = None
    dr_grade_right: int | None = None
    dr_confidence_left: float | None = None
    dr_confidence_right: float | None = None
    glaucoma_prob_left: float | None = None
    glaucoma_prob_right: float | None = None
    amd_prob_left: float | None = None
    amd_prob_right: float | None = None
    overall_risk: str | None = None
    referral_required: bool | None = None
    referral_urgency: str | None = None
    referral_reason: str | None = None
    notes: str | None = None
    raw_results: dict | None = None
    screened_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ScreeningListResponse(BaseModel):
    items: list[ScreeningResponse]
    total: int
    page: int
    page_size: int


class AnalysisRequest(BaseModel):
    """Optional parameters for the analysis trigger."""
    models: list[str] = Field(
        default=["dr", "glaucoma", "amd"],
        description="Which models to run",
    )
    generate_gradcam: bool = True
    generate_report: bool = True


class UploadResponse(BaseModel):
    image_id: UUID
    eye: str
    iqa_score: float
    iqa_passed: bool
    message: str


class AnalysisResponse(BaseModel):
    screening_id: UUID
    status: str
    message: str
    task_id: str | None = None  # Celery task ID if async
