"""FastAPI application entry point with lifespan, CORS, and router mounting."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.middleware.error_handler import ErrorHandlerMiddleware
from server.middleware.logging import RequestLoggingMiddleware
from server.middleware.rate_limit import RateLimitMiddleware
from server.middleware.request_id import RequestIDMiddleware

from server.api import health, auth, screenings, images, reports, patients, stores, webhooks
from server.api import demo, report_demo
from server.api import ws as ws_router

logger = logging.getLogger(__name__)

# Resolve checkpoint path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DR_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dr_aptos" / "best.pth"


def _pick_device() -> str:
    """Pick the best available device."""
    device_override = os.environ.get("MODEL_DEVICE", settings.MODEL_DEVICE)
    if device_override and device_override != "auto":
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database, Redis cache, and ML models on startup; clean up on shutdown."""
    logger.info("Starting NetraAI server ...")

    # ── Initialize database (create tables) ──────────────────────────
    from server.database import init_db, close_db

    try:
        await init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e, exc_info=True)
        logger.warning("Server will start WITHOUT database connectivity.")

    # ── Initialize Redis cache ───────────────────────────────────────
    from server.services.cache import cache

    try:
        await cache.connect()
    except Exception as e:
        logger.warning("Redis cache initialization failed (non-fatal): %s", e)

    # ── Load EfficientNet-B3 DR model (the one we actually trained) ──
    from server.services.inference_v2 import InferenceService

    inference_svc = InferenceService()

    if DR_CHECKPOINT.exists():
        device = _pick_device()
        try:
            await inference_svc.load_model(str(DR_CHECKPOINT), device=device)
            logger.info("DR model loaded on %s from %s", device, DR_CHECKPOINT)
        except Exception as e:
            logger.error("Failed to load DR model: %s", e, exc_info=True)
            logger.warning("Server will start WITHOUT the DR model.")
    else:
        logger.warning(
            "DR checkpoint not found at %s — server will start without inference.",
            DR_CHECKPOINT,
        )

    app.state.inference_service = inference_svc

    # ── Optionally try to load the old ONNX ModelRegistry (graceful) ─
    try:
        from server.services.inference import ModelRegistry

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
        logger.info("Legacy ModelRegistry loaded (ONNX models).")
    except Exception as e:
        logger.warning("Legacy ModelRegistry not available: %s (this is OK)", e)
        app.state.models = None

    logger.info("NetraAI server ready.")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources...")

    # Close Redis cache
    try:
        await cache.close()
    except Exception:
        pass

    # Release ML models
    await inference_svc.unload_model()
    if getattr(app.state, "models", None) is not None:
        try:
            await app.state.models.unload_models()
        except Exception:
            pass

    # Close database pool
    try:
        await close_db()
    except Exception:
        pass

    logger.info("All resources released. Goodbye.")


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
    # Note: middleware executes in reverse registration order (last added = outermost).
    # RequestID is outermost so all other middleware can use request_id.
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
    app.include_router(screenings.router, prefix="/v1/screenings", tags=["screenings"])
    app.include_router(images.router, prefix="/v1/images", tags=["images"])
    app.include_router(reports.router, prefix="/v1/reports", tags=["reports"])
    app.include_router(patients.router, prefix="/v1/patients", tags=["patients"])
    app.include_router(stores.router, prefix="/v1/stores", tags=["stores"])
    app.include_router(webhooks.router, prefix="/v1/webhooks", tags=["webhooks"])
    app.include_router(demo.router, prefix="/v1/demo", tags=["demo"])
    app.include_router(report_demo.router, prefix="/v1/demo", tags=["demo-report"])

    # ── WebSocket routes ─────────────────────────────────────────────
    app.include_router(ws_router.router, prefix="/v1/ws", tags=["websocket"])

    return app


app = create_app()
