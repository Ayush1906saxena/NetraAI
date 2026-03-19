"""
Global exception handler middleware for production-grade error handling.

Catches all unhandled exceptions, logs them with full context, and returns
structured JSON error responses. Never exposes internal details in production.
"""

import logging
import time
import traceback
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import settings

logger = logging.getLogger("netra.errors")


# ── Mapping of known exception types to HTTP status codes ────────────────
_EXCEPTION_STATUS_MAP: dict[type, int] = {
    ValueError: 400,
    TypeError: 400,
    FileNotFoundError: 404,
    PermissionError: 403,
    NotImplementedError: 501,
}

# User-safe messages per status code (used in production)
_SAFE_MESSAGES: dict[int, str] = {
    400: "Bad request. Please check your input and try again.",
    403: "You do not have permission to perform this action.",
    404: "The requested resource was not found.",
    500: "An internal server error occurred. Please try again later.",
    501: "This feature is not yet implemented.",
    503: "Service temporarily unavailable. Please try again shortly.",
}


def _is_production() -> bool:
    return settings.ENVIRONMENT.lower() in ("production", "prod", "staging")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Catches all unhandled exceptions that escape route handlers, logs them
    with full traceback and request context, and returns a structured JSON
    error response.  In production, internal details are redacted.
    """

    async def dispatch(self, request: Request, call_next):
        # Ensure a request_id exists (may already be set by RequestLoggingMiddleware)
        request_id = getattr(request.state, "request_id", None)
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
            request.state.request_id = request_id

        start = time.perf_counter()

        try:
            response = await call_next(request)
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            status_code = _EXCEPTION_STATUS_MAP.get(type(exc), 500)

            # Build structured log context
            log_extra = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 1),
                "client_ip": request.client.host if request.client else "unknown",
                "exception_type": type(exc).__name__,
            }

            # Log at appropriate level with full traceback
            if status_code >= 500:
                logger.error(
                    "[%s] Unhandled %s on %s %s -> %d (%.1fms): %s",
                    request_id,
                    type(exc).__name__,
                    request.method,
                    request.url.path,
                    status_code,
                    duration_ms,
                    str(exc),
                    exc_info=True,
                    extra=log_extra,
                )
            else:
                logger.warning(
                    "[%s] %s on %s %s -> %d (%.1fms): %s",
                    request_id,
                    type(exc).__name__,
                    request.method,
                    request.url.path,
                    status_code,
                    duration_ms,
                    str(exc),
                    extra=log_extra,
                )

            # Build client-facing response
            if _is_production():
                detail = _SAFE_MESSAGES.get(status_code, _SAFE_MESSAGES[500])
            else:
                # In development, include the actual error message (but not full traceback)
                detail = str(exc) or _SAFE_MESSAGES.get(status_code, _SAFE_MESSAGES[500])

            body = {
                "error": {
                    "status": status_code,
                    "message": detail,
                    "request_id": request_id,
                },
            }

            # In development mode, attach exception type and traceback summary
            if not _is_production():
                body["error"]["type"] = type(exc).__name__
                body["error"]["traceback"] = traceback.format_exception_only(
                    type(exc), exc
                )

            return JSONResponse(
                status_code=status_code,
                content=body,
                headers={
                    "X-Request-ID": request_id,
                    "X-Response-Time": f"{duration_ms:.1f}ms",
                },
            )
