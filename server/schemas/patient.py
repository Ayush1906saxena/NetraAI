"""Pydantic schemas for patient CRUD."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    date_of_birth: date | None = None
    age: int | None = Field(None, ge=0, le=150)
    gender: str | None = Field(None, pattern=r"^(male|female|other)$")
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    abha_id: str | None = None
    abha_address: str | None = None
    is_diabetic: bool | None = None
    diabetes_duration_years: int | None = None
    has_hypertension: bool | None = None
    medical_notes: str | None = None


class PatientUpdate(BaseModel):
    full_name: str | None = None
    date_of_birth: date | None = None
    age: int | None = None
    gender: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    abha_id: str | None = None
    abha_address: str | None = None
    is_diabetic: bool | None = None
    diabetes_duration_years: int | None = None
    has_hypertension: bool | None = None
    medical_notes: str | None = None


class PatientResponse(BaseModel):
    id: UUID
    full_name: str
    date_of_birth: date | None = None
    age: int | None = None
    gender: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    abha_id: str | None = None
    abha_address: str | None = None
    is_diabetic: bool | None = None
    diabetes_duration_years: int | None = None
    has_hypertension: bool | None = None
    medical_notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PatientListResponse(BaseModel):
    items: list[PatientResponse]
    total: int
    page: int
    page_size: int
