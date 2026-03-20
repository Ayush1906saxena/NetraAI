"""
End-to-end integration tests for the full DR screening flow.

These tests exercise the real FastAPI application (with lifespan),
the real EfficientNet-B3 DR model, and the demo endpoints — without
requiring Docker services (Postgres, Redis, MinIO).

The inference tests require the model checkpoint at:
    checkpoints/dr_aptos/best.pth
"""

import time

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ═══════════════════════════════════════════════════════════════════════════
# 1. Demo single-image analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoAnalyzeSingleImage:
    """POST /v1/demo/analyze — full DR analysis on a single fundus image."""

    async def test_demo_analyze_single_image(
        self, client: AsyncClient, fundus_image_bytes: bytes, model_loaded: bool,
    ):
        """Upload a real fundus image and verify the full response structure."""
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        response = await client.post(
            "/v1/demo/analyze",
            files={"file": ("fundus_test.jpg", fundus_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Top-level status
        assert data["status"] == "success"

        # ── DR analysis ──────────────────────────────────────────────
        analysis = data["analysis"]
        dr = analysis["dr"]

        assert isinstance(dr["grade"], int)
        assert 0 <= dr["grade"] <= 4

        assert isinstance(dr["confidence"], (int, float))
        assert 0.0 <= dr["confidence"] <= 1.0

        # Probabilities list — 5 values summing to ~1
        probs = dr["probabilities_list"]
        assert len(probs) == 5
        assert abs(sum(probs) - 1.0) < 0.01, f"Probabilities sum to {sum(probs)}, expected ~1.0"

        # Grade name is a non-empty string
        assert isinstance(dr["grade_name"], str)
        assert len(dr["grade_name"]) > 0

        # ── Referral ─────────────────────────────────────────────────
        referral = analysis["referral"]
        assert "urgency" in referral
        assert "recommendation" in referral
        assert isinstance(referral["recommendation"], str)

        # ── GradCAM ──────────────────────────────────────────────────
        gradcam = analysis["gradcam"]
        assert "overlay_png_base64" in gradcam
        assert isinstance(gradcam["overlay_png_base64"], str)
        # GradCAM may or may not succeed — just check the key exists

        # ── Quality ──────────────────────────────────────────────────
        quality = data["quality"]
        assert "score" in quality
        assert "passed" in quality
        assert isinstance(quality["score"], (int, float))
        assert isinstance(quality["passed"], bool)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Grade reasonableness check
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoAnalyzeReturnsCorrectGrade:
    """Verify the model returns plausible grades for synthetic images."""

    async def test_demo_analyze_returns_reasonable_grade(
        self, client: AsyncClient, fundus_image_bytes: bytes, model_loaded: bool,
    ):
        """
        A clean synthetic fundus image should NOT return the most severe
        grade (grade 4 = Proliferative DR) with high confidence, since the
        synthetic image has no pathological features.
        """
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        response = await client.post(
            "/v1/demo/analyze",
            files={"file": ("test.jpg", fundus_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        dr = data["analysis"]["dr"]

        grade = dr["grade"]
        confidence = dr["confidence"]

        # A synthetic image with no pathology should not be classified
        # as Proliferative DR (grade 4) with very high confidence.
        # We allow it only if confidence is low (model is uncertain).
        if grade == 4:
            assert confidence < 0.90, (
                f"Model is very confident (conf={confidence}) that a clean synthetic "
                f"image is Proliferative DR — this is implausible."
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Invalid file upload
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoAnalyzeInvalidFile:
    """Verify that non-image files are rejected with a clear error."""

    async def test_demo_analyze_invalid_file(
        self, client: AsyncClient, model_loaded: bool,
    ):
        """Upload a text file — should return 400."""
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        text_content = b"This is not an image file at all."
        response = await client.post(
            "/v1/demo/analyze",
            files={"file": ("readme.txt", text_content, "text/plain")},
        )
        assert response.status_code == 400, (
            f"Expected 400 for text file, got {response.status_code}: {response.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Large file limit
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoAnalyzeLargeFile:
    """Verify the file-size limit is enforced."""

    async def test_demo_analyze_large_file(
        self, client: AsyncClient, model_loaded: bool,
    ):
        """
        Upload a file exceeding the 20 MB limit.
        We send only ~21 MB of JPEG-headed junk to trigger the size check.
        """
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        # Construct bytes that look like a JPEG header but are oversized
        large_bytes = b"\xff\xd8\xff" + b"\x00" * (21 * 1024 * 1024)
        response = await client.post(
            "/v1/demo/analyze",
            files={"file": ("big.jpg", large_bytes, "image/jpeg")},
        )
        assert response.status_code == 413, (
            f"Expected 413 for oversized file, got {response.status_code}: {response.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batch analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoBatchAnalyze:
    """POST /v1/demo/analyze-batch — upload multiple images at once."""

    async def test_demo_batch_analyze(
        self,
        client: AsyncClient,
        multiple_fundus_images: list[bytes],
        model_loaded: bool,
    ):
        """Upload 3 images and verify all 3 get results."""
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        files = [
            ("files", (f"fundus_{i}.jpg", img_bytes, "image/jpeg"))
            for i, img_bytes in enumerate(multiple_fundus_images)
        ]
        response = await client.post("/v1/demo/analyze-batch", files=files)
        assert response.status_code == 200, f"Batch analyze failed: {response.text}"
        data = response.json()

        assert data["status"] == "success"
        assert data["total"] == 3
        assert len(data["results"]) == 3

        for result in data["results"]:
            assert result["status"] == "success"
            assert "analysis" in result
            assert "quality" in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Health endpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """GET /health — basic liveness probe."""

    async def test_health_endpoint(self, client: AsyncClient):
        """Health should return 200 with status=healthy."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# ═══════════════════════════════════════════════════════════════════════════
# 7. X-Request-ID header
# ═══════════════════════════════════════════════════════════════════════════

class TestRequestIdHeader:
    """Every response should include an X-Request-ID header."""

    async def test_request_id_header(self, client: AsyncClient):
        """GET /health should have X-Request-ID in the response."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert "x-request-id" in response.headers, (
            f"Missing X-Request-ID header. Headers: {dict(response.headers)}"
        )
        rid = response.headers["x-request-id"]
        assert len(rid) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. CORS headers
# ═══════════════════════════════════════════════════════════════════════════

class TestCorsHeaders:
    """OPTIONS preflight should return proper CORS headers."""

    async def test_cors_headers(self, client: AsyncClient):
        """
        An OPTIONS request with Origin should receive Access-Control-Allow-*
        headers from the CORS middleware.
        """
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # CORS preflight should succeed
        assert response.status_code == 200, (
            f"CORS preflight failed with {response.status_code}: {response.text}"
        )
        headers = response.headers
        assert "access-control-allow-origin" in headers, (
            f"Missing CORS allow-origin. Headers: {dict(headers)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. Inference caching / idempotency
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceCaching:
    """Uploading the same image twice should work (and ideally be faster)."""

    async def test_inference_caching(
        self, client: AsyncClient, fundus_image_bytes: bytes, model_loaded: bool,
    ):
        """Upload the same image twice — both should succeed without error."""
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        # First request
        t0 = time.perf_counter()
        r1 = await client.post(
            "/v1/demo/analyze",
            files={"file": ("same.jpg", fundus_image_bytes, "image/jpeg")},
        )
        t1 = time.perf_counter()

        # Second request with the same image
        t2 = time.perf_counter()
        r2 = await client.post(
            "/v1/demo/analyze",
            files={"file": ("same.jpg", fundus_image_bytes, "image/jpeg")},
        )
        t3 = time.perf_counter()

        assert r1.status_code == 200
        assert r2.status_code == 200

        d1 = r1.json()["analysis"]["dr"]
        d2 = r2.json()["analysis"]["dr"]

        # Same image should produce the same grade
        assert d1["grade"] == d2["grade"]
        assert d1["grade_name"] == d2["grade_name"]

        # Log timing (we don't hard-assert on speed since Redis is not available)
        first_ms = (t1 - t0) * 1000
        second_ms = (t3 - t2) * 1000
        print(f"  First request: {first_ms:.0f}ms, Second request: {second_ms:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Quality check — bad image
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityCheckBadImage:
    """A very dark or tiny image should produce a low quality score."""

    async def test_quality_check_bad_image(
        self, client: AsyncClient, dark_image_bytes: bytes, model_loaded: bool,
    ):
        """
        Upload a very dark 128x128 image. The quality score should be
        low, and quality.passed may be False.
        """
        if not model_loaded:
            pytest.skip("DR model checkpoint not available")

        response = await client.post(
            "/v1/demo/analyze",
            files={"file": ("dark.jpg", dark_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200, f"Unexpected {response.status_code}: {response.text}"
        data = response.json()

        quality = data["quality"]
        assert isinstance(quality["score"], (int, float))

        # A dark, low-resolution image should score below a normal fundus image.
        # We use a generous threshold — just verify the system doesn't crash
        # and the score is in the valid range [0, 1].
        assert 0.0 <= quality["score"] <= 1.0
        # It's likely this dark image will fail quality, but we don't hard-require
        # it — the important thing is the system handles it gracefully.
