"""
Redis caching layer for prediction results and screening data.

Gracefully falls back to no-op if Redis is unavailable — logs a warning
and continues without caching so the application never hard-fails on cache issues.
"""

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as redis

from server.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache with graceful degradation."""

    def __init__(self, redis_url: str | None = None):
        self._url = redis_url or settings.REDIS_URL
        self._client: redis.Redis | None = None
        self._available: bool = False

    async def connect(self) -> None:
        """Establish the Redis connection. Safe to call multiple times."""
        try:
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Verify connectivity
            await self._client.ping()
            self._available = True
            logger.info("Redis cache connected at %s", self._url)
        except Exception as exc:
            logger.warning(
                "Redis unavailable at %s — caching disabled: %s", self._url, exc
            )
            self._available = False
            self._client = None

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._available = False
            logger.info("Redis cache connection closed.")

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """Compute SHA-256 hex digest for image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    async def _get(self, key: str) -> str | None:
        if not self.is_available:
            return None
        try:
            return await self._client.get(key)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Redis GET failed for key=%s: %s", key, exc)
            return None

    async def _set(self, key: str, value: str, ttl: int = 3600) -> None:
        if not self.is_available:
            return
        try:
            await self._client.set(key, value, ex=ttl)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Redis SET failed for key=%s: %s", key, exc)

    # ── Prediction cache ──────────────────────────────────────────────────

    async def get_prediction_cache(self, image_hash: str) -> dict[str, Any] | None:
        """Retrieve a cached prediction result by image hash.

        Returns:
            Deserialized dict if found, else None.
        """
        raw = await self._get(f"pred:{image_hash}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_prediction_cache(
        self,
        image_hash: str,
        result: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """Cache a prediction result keyed by image hash."""
        try:
            payload = json.dumps(result, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Cannot serialize prediction for caching: %s", exc)
            return
        await self._set(f"pred:{image_hash}", payload, ttl=ttl)

    # ── Screening cache ───────────────────────────────────────────────────

    async def get_screening_cache(self, screening_id: str) -> dict[str, Any] | None:
        """Retrieve cached screening data."""
        raw = await self._get(f"screening:{screening_id}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_screening_cache(
        self,
        screening_id: str,
        data: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """Cache screening data."""
        try:
            payload = json.dumps(data, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Cannot serialize screening for caching: %s", exc)
            return
        await self._set(f"screening:{screening_id}", payload, ttl=ttl)


# ── Module-level singleton ────────────────────────────────────────────────
cache = RedisCache()
