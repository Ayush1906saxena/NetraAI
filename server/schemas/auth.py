"""Pydantic schemas for authentication."""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=255)
    phone: str | None = Field(None, pattern=r"^\+?[1-9]\d{6,14}$")
    role: str = Field("operator", pattern=r"^(admin|doctor|operator|store_manager)$")
    store_id: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str
    email: str
    role: str
    store_id: str | None = None
    exp: int


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    phone: str | None = None
    role: str
    store_id: str | None = None
    is_active: bool
    is_verified: bool

    model_config = {"from_attributes": True}


class RefreshRequest(BaseModel):
    refresh_token: str
