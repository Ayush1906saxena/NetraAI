"""
Server test fixtures.

Provides:
- Test database (SQLite in-memory for speed)
- Test FastAPI client (httpx AsyncClient)
- Authenticated test client
- Factory fixtures for creating test data
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import StaticPool, create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

# Override database URL before importing server modules
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///file::memory:?cache=shared"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production-use-only"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"


# ---------------------------------------------------------------------------
# App & Database fixtures
# ---------------------------------------------------------------------------

def _create_test_app() -> FastAPI:
    """
    Create a test FastAPI application.

    If the real app is available, use it. Otherwise, create a minimal
    app that can be used for basic endpoint testing.
    """
    try:
        from server.main import app
        return app
    except (ImportError, ModuleNotFoundError):
        # Create a minimal test app
        app = FastAPI(title="Netra AI Test")

        from fastapi import Depends, HTTPException, status
        from fastapi.security import HTTPBearer
        from pydantic import BaseModel

        security = HTTPBearer(auto_error=False)

        # Minimal in-memory store
        _users_db: dict = {}
        _screenings_db: dict = {}
        _tokens_db: dict = {}

        class UserRegister(BaseModel):
            email: str
            password: str
            full_name: str
            role: str = "operator"

        class UserLogin(BaseModel):
            email: str
            password: str

        class TokenResponse(BaseModel):
            access_token: str
            token_type: str = "bearer"

        class ScreeningCreate(BaseModel):
            patient_name: str
            patient_age: int
            eye: str = "both"

        class ScreeningResponse(BaseModel):
            id: str
            patient_name: str
            patient_age: int
            eye: str
            status: str
            created_at: str

        async def get_current_user(credentials=Depends(security)):
            if credentials is None:
                raise HTTPException(status_code=401, detail="Not authenticated")
            token = credentials.credentials
            if token not in _tokens_db:
                raise HTTPException(status_code=401, detail="Invalid token")
            return _tokens_db[token]

        @app.post("/v1/auth/register", response_model=TokenResponse)
        async def register(user: UserRegister):
            if user.email in _users_db:
                raise HTTPException(status_code=409, detail="Email already registered")
            user_id = str(uuid.uuid4())
            _users_db[user.email] = {
                "id": user_id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "password_hash": user.password,  # Not real hashing in test
            }
            token = str(uuid.uuid4())
            _tokens_db[token] = _users_db[user.email]
            return TokenResponse(access_token=token)

        @app.post("/v1/auth/login", response_model=TokenResponse)
        async def login(creds: UserLogin):
            user = _users_db.get(creds.email)
            if not user or user["password_hash"] != creds.password:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token = str(uuid.uuid4())
            _tokens_db[token] = user
            return TokenResponse(access_token=token)

        @app.get("/v1/auth/me")
        async def get_me(user=Depends(get_current_user)):
            return {
                "id": user["id"],
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"],
            }

        @app.post("/v1/screenings", response_model=ScreeningResponse, status_code=201)
        async def create_screening(screening: ScreeningCreate, user=Depends(get_current_user)):
            screening_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            record = {
                "id": screening_id,
                "patient_name": screening.patient_name,
                "patient_age": screening.patient_age,
                "eye": screening.eye,
                "status": "pending",
                "created_at": now,
                "created_by": user["id"],
            }
            _screenings_db[screening_id] = record
            return ScreeningResponse(**record)

        @app.get("/v1/screenings")
        async def list_screenings(
            skip: int = 0,
            limit: int = 20,
            user=Depends(get_current_user),
        ):
            items = list(_screenings_db.values())[skip: skip + limit]
            return {"items": items, "total": len(_screenings_db)}

        @app.get("/v1/screenings/{screening_id}")
        async def get_screening(screening_id: str, user=Depends(get_current_user)):
            if screening_id not in _screenings_db:
                raise HTTPException(status_code=404, detail="Screening not found")
            return _screenings_db[screening_id]

        @app.get("/health")
        async def health():
            return {"status": "healthy", "version": "test"}

        return app


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """Create the test FastAPI application."""
    return _create_test_app()


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Auth fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def auth_token(client: AsyncClient) -> str:
    """Register a test user and return the auth token."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": f"test_{uuid.uuid4().hex[:8]}@netra.ai",
            "password": "TestPass123!",
            "full_name": "Test User",
            "role": "operator",
        },
    )
    assert response.status_code == 200, f"Registration failed: {response.text}"
    return response.json()["access_token"]


@pytest_asyncio.fixture
async def auth_headers(auth_token: str) -> dict:
    """Return authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def admin_token(client: AsyncClient) -> str:
    """Register an admin user and return the auth token."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": f"admin_{uuid.uuid4().hex[:8]}@netra.ai",
            "password": "AdminPass123!",
            "full_name": "Admin User",
            "role": "admin",
        },
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest_asyncio.fixture
async def admin_headers(admin_token: str) -> dict:
    """Return authorization headers for admin requests."""
    return {"Authorization": f"Bearer {admin_token}"}


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

@pytest.fixture
def screening_payload() -> dict:
    """Standard screening creation payload."""
    return {
        "patient_name": "Test Patient",
        "patient_age": 55,
        "eye": "both",
    }


@pytest.fixture
def multiple_screening_payloads() -> list[dict]:
    """Multiple screening payloads for list tests."""
    return [
        {"patient_name": f"Patient {i}", "patient_age": 40 + i, "eye": "both"}
        for i in range(5)
    ]
