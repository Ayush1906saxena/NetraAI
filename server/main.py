"""FastAPI application entry point with lifespan, CORS, and router mounting."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.services.inference import ModelRegistry
from server.middleware.logging import RequestLoggingMiddleware
from server.middleware.rate_limit import RateLimitMiddleware

from server.api import health, auth, screenings, images, reports, patients, stores, webhooks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, release on shutdown."""
    logger.info("Starting NetraAI server — loading ML models...")
    registry = ModelRegistry()
    await registry.load_models(
        dr_path=settings.MODEL_DR_PATH,
        glaucoma_path=settings.MODEL_GLAUCOMA_PATH,
        amd_path=settings.MODEL_AMD_PATH,
        iqa_path=settings.MODEL_IQA_PATH,
        segmentation_path=settings.MODEL_SEGMENTATION_PATH,
        device=settings.MODEL_DEVICE,
    )
    app.state.models = registry
    logger.info("All ML models loaded successfully.")
    yield
    logger.info("Shutting down — releasing ML models...")
    await registry.unload_models()
    logger.info("Models released. Goodbye.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Retinal screening AI platform",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # ── Custom middleware ─────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
    app.include_router(screenings.router, prefix="/v1/screenings", tags=["screenings"])
    app.include_router(images.router, prefix="/v1/images", tags=["images"])
    app.include_router(reports.router, prefix="/v1/reports", tags=["reports"])
    app.include_router(patients.router, prefix="/v1/patients", tags=["patients"])
    app.include_router(stores.router, prefix="/v1/stores", tags=["stores"])
    app.include_router(webhooks.router, prefix="/v1/webhooks", tags=["webhooks"])

    return app


app = create_app()
