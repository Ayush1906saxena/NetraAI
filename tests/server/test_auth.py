"""
Tests for authentication endpoints.

Covers:
- User registration
- Login
- Token-based access
- Protected endpoint access
- Duplicate registration
- Invalid credentials
"""

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


class TestRegistration:
    """Tests for POST /v1/auth/register."""

    async def test_register_success(self, client: AsyncClient):
        """Should register a new user and return a token."""
        response = await client.post(
            "/v1/auth/register",
            json={
                "email": f"newuser_{uuid.uuid4().hex[:8]}@netra.ai",
                "password": "SecurePass123!",
                "full_name": "New User",
                "role": "operator",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0

    async def test_register_returns_usable_token(self, client: AsyncClient):
        """The returned token should work for authenticated endpoints."""
        email = f"tokentest_{uuid.uuid4().hex[:8]}@netra.ai"
        reg_response = await client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": "SecurePass123!",
                "full_name": "Token Test",
                "role": "operator",
            },
        )
        token = reg_response.json()["access_token"]

        # Use the token
        me_response = await client.get(
            "/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["email"] == email

    async def test_register_duplicate_email(self, client: AsyncClient):
        """Should reject duplicate email registration."""
        email = f"dupe_{uuid.uuid4().hex[:8]}@netra.ai"
        payload = {
            "email": email,
            "password": "SecurePass123!",
            "full_name": "First User",
            "role": "operator",
        }

        # First registration should succeed
        response1 = await client.post("/v1/auth/register", json=payload)
        assert response1.status_code == 200

        # Second registration with same email should fail
        response2 = await client.post("/v1/auth/register", json=payload)
        assert response2.status_code == 409

    async def test_register_missing_fields(self, client: AsyncClient):
        """Should return 422 for missing required fields."""
        response = await client.post(
            "/v1/auth/register",
            json={"email": "test@netra.ai"},  # Missing password, full_name
        )
        assert response.status_code == 422

    async def test_register_with_role(self, client: AsyncClient):
        """Should accept operator and admin roles."""
        for role in ["operator", "admin"]:
            response = await client.post(
                "/v1/auth/register",
                json={
                    "email": f"{role}_{uuid.uuid4().hex[:8]}@netra.ai",
                    "password": "SecurePass123!",
                    "full_name": f"{role.title()} User",
                    "role": role,
                },
            )
            assert response.status_code == 200


class TestLogin:
    """Tests for POST /v1/auth/login."""

    async def test_login_success(self, client: AsyncClient):
        """Should login with correct credentials."""
        email = f"login_{uuid.uuid4().hex[:8]}@netra.ai"
        password = "SecurePass123!"

        # Register first
        await client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": "Login Test",
            },
        )

        # Login
        response = await client.post(
            "/v1/auth/login",
            json={"email": email, "password": password},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_wrong_password(self, client: AsyncClient):
        """Should reject wrong password."""
        email = f"wrongpw_{uuid.uuid4().hex[:8]}@netra.ai"

        await client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": "CorrectPass123!",
                "full_name": "Test",
            },
        )

        response = await client.post(
            "/v1/auth/login",
            json={"email": email, "password": "WrongPassword!"},
        )
        assert response.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Should reject login for non-existent email."""
        response = await client.post(
            "/v1/auth/login",
            json={
                "email": "nobody@netra.ai",
                "password": "anything",
            },
        )
        assert response.status_code == 401

    async def test_login_token_works(self, client: AsyncClient):
        """Token from login should work for authenticated endpoints."""
        email = f"logintoken_{uuid.uuid4().hex[:8]}@netra.ai"
        password = "SecurePass123!"

        await client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": "Login Token Test",
            },
        )

        login_response = await client.post(
            "/v1/auth/login",
            json={"email": email, "password": password},
        )
        token = login_response.json()["access_token"]

        me_response = await client.get(
            "/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["email"] == email


class TestProtectedAccess:
    """Tests for authentication enforcement on protected endpoints."""

    async def test_no_token_returns_401(self, client: AsyncClient):
        """Protected endpoints should return 401 without token."""
        response = await client.get("/v1/auth/me")
        assert response.status_code == 401

    async def test_invalid_token_returns_401(self, client: AsyncClient):
        """Should reject invalid tokens."""
        response = await client.get(
            "/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token-12345"},
        )
        assert response.status_code == 401

    async def test_malformed_auth_header(self, client: AsyncClient):
        """Should reject malformed authorization headers."""
        response = await client.get(
            "/v1/auth/me",
            headers={"Authorization": "NotBearer token123"},
        )
        # FastAPI's HTTPBearer will reject this
        assert response.status_code in (401, 403)

    async def test_get_me_returns_user_info(
        self, client: AsyncClient, auth_headers: dict
    ):
        """GET /v1/auth/me should return current user info."""
        response = await client.get("/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
        assert "full_name" in data
        assert "role" in data

    async def test_screenings_require_auth(self, client: AsyncClient):
        """Screening endpoints should require authentication."""
        # POST
        response = await client.post(
            "/v1/screenings",
            json={"patient_name": "Test", "patient_age": 50},
        )
        assert response.status_code == 401

        # GET list
        response = await client.get("/v1/screenings")
        assert response.status_code == 401

        # GET single
        response = await client.get("/v1/screenings/some-id")
        assert response.status_code == 401

    async def test_health_is_public(self, client: AsyncClient):
        """Health endpoint should be accessible without auth."""
        response = await client.get("/health")
        assert response.status_code == 200
