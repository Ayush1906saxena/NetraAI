"""Store CRUD endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.dependencies import get_current_user, get_db

from server.models.store import Store

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Inline schemas for stores ────────────────────────────────────────────

class StoreCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    code: str = Field(..., min_length=2, max_length=50)
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    phone: str | None = None
    email: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    manager_name: str | None = None
    equipment_info: str | None = None


class StoreUpdate(BaseModel):
    name: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    phone: str | None = None
    email: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    is_active: bool | None = None
    manager_name: str | None = None
    equipment_info: str | None = None


class StoreResponse(BaseModel):
    id: uuid.UUID
    name: str
    code: str
    address: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    phone: str | None = None
    email: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    is_active: bool
    manager_name: str | None = None
    equipment_info: str | None = None

    model_config = {"from_attributes": True}


class StoreListResponse(BaseModel):
    items: list[StoreResponse]
    total: int
    page: int
    page_size: int


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/", response_model=StoreResponse, status_code=status.HTTP_201_CREATED)
async def create_store(
    body: StoreCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new store."""
    existing = await db.execute(select(Store).where(Store.code == body.code))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Store code already exists")

    store = Store(id=uuid.uuid4(), **body.model_dump())
    db.add(store)
    await db.flush()
    await db.refresh(store)
    logger.info("Store created: %s (%s)", store.name, store.code)
    return store


@router.get("/", response_model=StoreListResponse)
async def list_stores(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    active_only: bool = True,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List stores with pagination."""
    query = select(Store)
    if active_only:
        query = query.where(Store.is_active == True)

    query = query.order_by(Store.name)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    items = result.scalars().all()

    return StoreListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{store_id}", response_model=StoreResponse)
async def get_store(
    store_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a store by ID."""
    result = await db.execute(select(Store).where(Store.id == store_id))
    store = result.scalar_one_or_none()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    return store


@router.put("/{store_id}", response_model=StoreResponse)
async def update_store(
    store_id: uuid.UUID,
    body: StoreUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update store details."""
    result = await db.execute(select(Store).where(Store.id == store_id))
    store = result.scalar_one_or_none()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(store, field, value)

    await db.flush()
    await db.refresh(store)
    logger.info("Store updated: %s", store_id)
    return store


@router.delete("/{store_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_store(
    store_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a store (deactivate)."""
    result = await db.execute(select(Store).where(Store.id == store_id))
    store = result.scalar_one_or_none()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    store.is_active = False
    await db.flush()
    logger.info("Store deactivated: %s", store_id)
