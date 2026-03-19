"""Per-store rate limiting middleware using in-memory sliding window."""

import logging
import time
from collections import defaultdict

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from server.config import settings

logger = logging.getLogger(__name__)


class _SlidingWindowCounter:
    """Simple in-memory sliding window rate limiter."""

    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        now = time.time()
        cutoff = now - window_seconds

        # Prune old entries
        self._windows[key] = [t for t in self._windows[key] if t > cutoff]

        if len(self._windows[key]) >= max_requests:
            return False

        self._windows[key].append(now)
        return True

    def remaining(self, key: str, max_requests: int, window_seconds: int = 60) -> int:
        now = time.time()
        cutoff = now - window_seconds
        self._windows[key] = [t for t in self._windows[key] if t > cutoff]
        return max(0, max_requests - len(self._windows[key]))


_limiter = _SlidingWindowCounter()

# Paths exempt from rate limiting
EXEMPT_PATHS = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate-limit requests per store or per IP."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in EXEMPT_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        # Determine rate limit key
        store_id = getattr(request.state, "store_id", None) if hasattr(request, "state") else None
        if store_id:
            key = f"store:{store_id}"
        else:
            key = f"ip:{request.client.host}" if request.client else "ip:unknown"

        max_per_minute = settings.RATE_LIMIT_PER_MINUTE

        if not _limiter.is_allowed(key, max_per_minute, window_seconds=60):
            remaining = 0
            logger.warning("Rate limit exceeded for %s on %s", key, path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(max_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        remaining = _limiter.remaining(key, max_per_minute)
        response.headers["X-RateLimit-Limit"] = str(max_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
