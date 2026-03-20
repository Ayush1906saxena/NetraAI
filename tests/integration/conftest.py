"""
Integration test fixtures.

Provides:
- Test FastAPI client (httpx AsyncClient) wired to the real app
- Synthetic fundus image bytes for upload testing
- Multiple test images of varying quality
- Fixtures that work WITHOUT Docker services (no Postgres/Redis/MinIO)
"""

import io
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Override environment variables BEFORE importing any server module ──────
# Use SQLite so no Postgres is needed
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///file::memory:?cache=shared"
os.environ["SECRET_KEY"] = "test-secret-key-integration"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["MODEL_DEVICE"] = "cpu"


# ---------------------------------------------------------------------------
# Helpers to generate synthetic fundus images
# ---------------------------------------------------------------------------

def _make_fundus_image(size: int = 512, seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic fundus-like RGB image.

    Creates a bright circular region on a dark background with colour
    variation and noise, roughly mimicking a retinal photograph.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    center = size // 2
    radius = size // 3
    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    img[mask, 0] = rng.randint(120, 200)  # Red
    img[mask, 1] = rng.randint(60, 120)   # Green
    img[mask, 2] = rng.randint(30, 80)    # Blue

    noise = rng.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    """Encode an RGB numpy array as JPEG bytes."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


def _encode_png(img: np.ndarray) -> bytes:
    """Encode an RGB numpy array as PNG bytes."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok, "PNG encoding failed"
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Application fixture — creates the real FastAPI app with lifespan
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def test_app():
    """
    Build the real FastAPI app and run its lifespan (startup/shutdown).

    This loads the DR model if the checkpoint exists, initializes the
    database (SQLite in-memory), and gracefully skips Redis/MinIO.
    """
    from server.main import create_app

    app = create_app()

    # Manually trigger lifespan
    async with app.router.lifespan_context(app):
        yield app


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP test client wired to the real app."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fundus_image_bytes() -> bytes:
    """A synthetic 512x512 fundus image encoded as JPEG."""
    img = _make_fundus_image(size=512, seed=42)
    return _encode_jpeg(img)


@pytest.fixture(scope="session")
def fundus_image_png_bytes() -> bytes:
    """A synthetic 512x512 fundus image encoded as PNG."""
    img = _make_fundus_image(size=512, seed=42)
    return _encode_png(img)


@pytest.fixture(scope="session")
def dark_image_bytes() -> bytes:
    """A very dark 256x256 image — should score poorly on quality."""
    rng = np.random.RandomState(99)
    img = rng.randint(0, 15, (256, 256, 3), dtype=np.uint8)
    return _encode_jpeg(img, quality=95)


@pytest.fixture(scope="session")
def small_bright_image_bytes() -> bytes:
    """A bright but small (64x64) image."""
    rng = np.random.RandomState(7)
    img = rng.randint(200, 255, (64, 64, 3), dtype=np.uint8)
    return _encode_jpeg(img, quality=70)


@pytest.fixture(scope="session")
def multiple_fundus_images(fundus_image_bytes) -> list[bytes]:
    """Three distinct synthetic fundus images for batch testing."""
    images = [fundus_image_bytes]
    for seed in (100, 200):
        img = _make_fundus_image(size=512, seed=seed)
        images.append(_encode_jpeg(img))
    return images


@pytest.fixture(scope="session")
def model_loaded(test_app) -> bool:
    """Return True if the DR model was successfully loaded during startup."""
    svc = getattr(test_app.state, "inference_service", None)
    return svc is not None and svc.is_loaded
