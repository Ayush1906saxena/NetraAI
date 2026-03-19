"""Celery application setup with Redis broker."""

from celery import Celery
from server.config import settings

celery_app = Celery(
    "netra_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "server.workers.analyze",
        "server.workers.notify",
        "server.workers.abdm_push",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=300,
    task_time_limit=600,
    result_expires=86400,
)
