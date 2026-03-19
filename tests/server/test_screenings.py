"""
Tests for screening endpoints.

Covers the core screening workflow:
- Create a screening
- List screenings
- Get screening by ID
- Handle missing screenings
- Pagination
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


class TestCreateScreening:
    """Tests for POST /v1/screenings."""

    async def test_create_screening_success(
        self, client: AsyncClient, auth_headers: dict, screening_payload: dict
    ):
        """Should create a screening and return 201."""
        response = await client.post(
            "/v1/screenings",
            json=screening_payload,
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["patient_name"] == screening_payload["patient_name"]
        assert data["patient_age"] == screening_payload["patient_age"]
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data

    async def test_create_screening_returns_id(
        self, client: AsyncClient, auth_headers: dict, screening_payload: dict
    ):
        """Returned screening should have a valid UUID-like ID."""
        response = await client.post(
            "/v1/screenings",
            json=screening_payload,
            headers=auth_headers,
        )
        assert response.status_code == 201
        screening_id = response.json()["id"]
        assert len(screening_id) > 0

    async def test_create_screening_requires_auth(
        self, client: AsyncClient, screening_payload: dict
    ):
        """Should return 401 without authentication."""
        response = await client.post(
            "/v1/screenings",
            json=screening_payload,
        )
        assert response.status_code == 401

    async def test_create_screening_invalid_payload(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Should return 422 for invalid payload."""
        response = await client.post(
            "/v1/screenings",
            json={},  # Missing required fields
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_create_screening_with_eye_options(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Should accept different eye values."""
        for eye in ["both", "left", "right"]:
            payload = {
                "patient_name": "Test",
                "patient_age": 50,
                "eye": eye,
            }
            response = await client.post(
                "/v1/screenings",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 201
            assert response.json()["eye"] == eye


class TestListScreenings:
    """Tests for GET /v1/screenings."""

    async def test_list_screenings_empty(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Should return empty list initially."""
        response = await client.get(
            "/v1/screenings",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    async def test_list_screenings_after_creation(
        self,
        client: AsyncClient,
        auth_headers: dict,
        multiple_screening_payloads: list[dict],
    ):
        """Should list all created screenings."""
        # Create several screenings
        created_ids = []
        for payload in multiple_screening_payloads:
            response = await client.post(
                "/v1/screenings",
                json=payload,
                headers=auth_headers,
            )
            assert response.status_code == 201
            created_ids.append(response.json()["id"])

        # List them
        response = await client.get(
            "/v1/screenings",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= len(created_ids)

    async def test_list_screenings_requires_auth(self, client: AsyncClient):
        """Should return 401 without authentication."""
        response = await client.get("/v1/screenings")
        assert response.status_code == 401

    async def test_list_screenings_pagination(
        self,
        client: AsyncClient,
        auth_headers: dict,
        multiple_screening_payloads: list[dict],
    ):
        """Should support skip and limit parameters."""
        # Create screenings
        for payload in multiple_screening_payloads:
            await client.post(
                "/v1/screenings",
                json=payload,
                headers=auth_headers,
            )

        # Paginate
        response = await client.get(
            "/v1/screenings?skip=0&limit=2",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 2


class TestGetScreening:
    """Tests for GET /v1/screenings/{id}."""

    async def test_get_screening_by_id(
        self, client: AsyncClient, auth_headers: dict, screening_payload: dict
    ):
        """Should retrieve a specific screening by ID."""
        # Create
        create_response = await client.post(
            "/v1/screenings",
            json=screening_payload,
            headers=auth_headers,
        )
        screening_id = create_response.json()["id"]

        # Get
        response = await client.get(
            f"/v1/screenings/{screening_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == screening_id
        assert data["patient_name"] == screening_payload["patient_name"]

    async def test_get_screening_not_found(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Should return 404 for non-existent screening."""
        response = await client.get(
            "/v1/screenings/nonexistent-id-12345",
            headers=auth_headers,
        )
        assert response.status_code == 404

    async def test_get_screening_requires_auth(self, client: AsyncClient):
        """Should return 401 without authentication."""
        response = await client.get("/v1/screenings/some-id")
        assert response.status_code == 401


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    async def test_health_check(self, client: AsyncClient):
        """Health check should be accessible without auth."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
