"""
Request ID middleware.

Generates a UUID for every incoming request and attaches it to:
- request.state.request_id (for use in handlers and other middleware)
- X-Request-ID response header (for client-side correlation)
- Structured logging context
"""

import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("netra.request_id")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID to every request for tracing/correlation.

    If the client sends an X-Request-ID header, it is reused; otherwise
    a new UUID4 is generated. The ID is always returned in the
    X-Request-ID response header.
    """

    async def dispatch(self, request: Request, call_next):
        # Reuse client-supplied ID or generate a new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Attach to request state so handlers/middleware can access it
        request.state.request_id = request_id

        logger.info(
            "[%s] %s %s",
            request_id,
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        # Always set the response header
        response.headers["X-Request-ID"] = request_id

        return response
