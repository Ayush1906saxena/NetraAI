"""JWT verification middleware — validates tokens on protected routes."""

import logging

from fastapi import Request, status
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from server.config import settings

logger = logging.getLogger(__name__)

# Paths that do not require authentication
PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/v1/auth/login",
    "/v1/auth/register",
    "/v1/auth/refresh",
    "/v1/webhooks/gupshup",
    "/v1/webhooks/abdm/callback",
    "/v1/webhooks/payment",
    "/v1/webhooks/test",
}

PUBLIC_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi",
    "/static",
)


class JWTVerificationMiddleware(BaseHTTPMiddleware):
    """Middleware that verifies JWT tokens on non-public routes."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip authentication for public paths
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        # OPTIONS requests (CORS preflight) pass through
        if request.method == "OPTIONS":
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]

        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            # Attach user info to request state for downstream use
            request.state.user_id = payload.get("sub")
            request.state.user_email = payload.get("email")
            request.state.user_role = payload.get("role")
            request.state.store_id = payload.get("store_id")
        except JWTError as e:
            logger.warning("JWT verification failed: %s", e)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
