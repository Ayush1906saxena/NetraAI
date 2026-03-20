"""Dependency injection functions for FastAPI endpoints."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from server.config import settings
from server.database import get_db  # noqa: F401 — re-export for existing imports

bearer_scheme = HTTPBearer(auto_error=False)


def get_models(request: Request):
    """Retrieve the legacy ModelRegistry stored in app state during lifespan."""
    registry = getattr(request.app.state, "models", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded yet",
        )
    return registry


def get_inference_service(request: Request):
    """Retrieve the InferenceService (EfficientNet-B3 DR model) from app state."""
    from server.services.inference_v2 import InferenceService

    svc: InferenceService | None = getattr(request.app.state, "inference_service", None)
    if svc is None or not svc.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DR inference model not loaded",
        )
    return svc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """Decode JWT and return the user payload dict."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return {
            "user_id": user_id,
            "email": payload.get("email", ""),
            "role": payload.get("role", "operator"),
            "store_id": payload.get("store_id"),
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(*allowed_roles: str):
    """Return a dependency that checks the current user has one of the allowed roles."""

    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user['role']}' not permitted. Required: {allowed_roles}",
            )
        return current_user

    return _check
