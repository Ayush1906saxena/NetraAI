"""Patient CRUD endpoints with ABHA ID linking."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_current_user, get_db
from server.models.patient import Patient
from server.schemas.patient import (
    PatientCreate,
    PatientListResponse,
    PatientResponse,
    PatientUpdate,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    body: PatientCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new patient record."""
    # Check duplicate ABHA
    if body.abha_id:
        existing = await db.execute(
            select(Patient).where(Patient.abha_id == body.abha_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Patient with this ABHA ID already exists",
            )

    patient = Patient(id=uuid.uuid4(), **body.model_dump(exclude_unset=True))
    db.add(patient)
    await db.flush()
    await db.refresh(patient)
    logger.info("Patient created: %s (ABHA: %s)", patient.id, patient.abha_id)
    return patient


@router.get("/", response_model=PatientListResponse)
async def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List patients with search and pagination."""
    query = select(Patient)

    if search:
        like = f"%{search}%"
        query = query.where(
            or_(
                Patient.full_name.ilike(like),
                Patient.phone.ilike(like),
                Patient.abha_id.ilike(like),
                Patient.email.ilike(like),
            )
        )

    query = query.order_by(Patient.created_at.desc())

    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar() or 0

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = result.scalars().all()

    return PatientListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a patient by ID."""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: uuid.UUID,
    body: PatientUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update patient details."""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)

    await db.flush()
    await db.refresh(patient)
    logger.info("Patient updated: %s", patient_id)
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a patient record."""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    await db.delete(patient)
    await db.flush()
    logger.info("Patient deleted: %s", patient_id)


@router.post("/{patient_id}/link-abha", response_model=PatientResponse)
async def link_abha(
    patient_id: uuid.UUID,
    abha_id: str = Query(..., min_length=1),
    abha_address: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Link an ABHA ID to an existing patient."""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Check uniqueness
    dup = await db.execute(
        select(Patient).where(Patient.abha_id == abha_id, Patient.id != patient_id)
    )
    if dup.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="ABHA ID already linked to another patient")

    patient.abha_id = abha_id
    if abha_address:
        patient.abha_address = abha_address

    await db.flush()
    await db.refresh(patient)
    logger.info("ABHA %s linked to patient %s", abha_id, patient_id)
    return patient
