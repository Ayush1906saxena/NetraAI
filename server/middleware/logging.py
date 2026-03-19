"""Request/response logging middleware."""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("netra.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""

        logger.info(
            "[%s] --> %s %s%s from %s",
            request_id,
            method,
            path,
            f"?{query}" if query else "",
            client_ip,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "[%s] <-- %s %s 500 %.1fms (unhandled: %s)",
                request_id,
                method,
                path,
                duration_ms,
                str(exc)[:200],
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code

        log_fn = logger.info if status_code < 400 else logger.warning if status_code < 500 else logger.error
        log_fn(
            "[%s] <-- %s %s %d %.1fms",
            request_id,
            method,
            path,
            status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        return response
