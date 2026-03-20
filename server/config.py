"""Application configuration using pydantic-settings."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "NetraAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://netra:netra@localhost:5432/netra_ai"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600

    # ── S3 / MinIO ───────────────────────────────────────────────────────
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_IMAGES: str = "netra-images"
    S3_BUCKET_REPORTS: str = "netra-reports"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False
    S3_PRESIGNED_EXPIRY: int = 3600

    # ── ML Model Paths ───────────────────────────────────────────────────
    MODEL_DR_PATH: str = "models/dr_classifier.onnx"
    MODEL_GLAUCOMA_PATH: str = "models/glaucoma_classifier.onnx"
    MODEL_AMD_PATH: str = "models/amd_classifier.onnx"
    MODEL_IQA_PATH: str = "models/iqa_model.onnx"
    MODEL_SEGMENTATION_PATH: str = "models/vessel_segmentation.onnx"
    MODEL_DEVICE: str = "cpu"
    MODEL_BATCH_SIZE: int = 4

    # ── Enhanced Inference Settings ───────────────────────────────────
    ENSEMBLE_MODELS: int = 3                     # Max DR ensemble snapshots
    TTA_ENABLED: bool = False                    # TTA off by default (speed)
    TTA_N_AUGMENTS: int = 5                      # TTA augmentation folds
    MC_DROPOUT_PASSES: int = 5                   # Monte Carlo Dropout passes
    TEMPERATURE_SCALING_DEFAULT: float = 1.5     # Default temperature for calibration
    IQA_QUALITY_THRESHOLD: float = 0.4           # Minimum IQA quality to proceed
    UNCERTAINTY_REVIEW_THRESHOLD: float = 0.15   # Std above this → needs review

    # ── JWT / Auth ───────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-in-production-super-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # ── Gupshup (WhatsApp / SMS) ─────────────────────────────────────────
    GUPSHUP_API_KEY: str = ""
    GUPSHUP_APP_NAME: str = ""
    GUPSHUP_BASE_URL: str = "https://api.gupshup.io/wa/api/v1"
    GUPSHUP_SOURCE_PHONE: str = ""
    GUPSHUP_SMS_URL: str = "https://enterprise.smsgupshup.com/GatewayAPI/rest"
    GUPSHUP_SMS_USERID: str = ""
    GUPSHUP_SMS_PASSWORD: str = ""

    # ── ABDM (Ayushman Bharat Digital Mission) ───────────────────────────
    ABDM_BASE_URL: str = "https://dev.abdm.gov.in/gateway"
    ABDM_CLIENT_ID: str = ""
    ABDM_CLIENT_SECRET: str = ""
    ABDM_AUTH_URL: str = "https://dev.abdm.gov.in/gateway/v0.5/sessions"
    ABDM_HIP_ID: str = ""
    ABDM_HIP_NAME: str = "NetraAI Eye Screening"
    ABDM_CALLBACK_URL: str = ""

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    # ── Rate Limiting ────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 100

    # ── Report Generation ────────────────────────────────────────────────
    REPORT_TEMPLATE_DIR: str = "templates/reports"
    REPORT_LOGO_PATH: str = "static/logo.png"


settings = Settings()
