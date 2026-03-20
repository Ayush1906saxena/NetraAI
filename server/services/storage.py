"""
MinIO / S3 storage operations using the minio Python library.

Provides async wrappers for uploading/downloading fundus images and
report PDFs. Auto-creates buckets on first use. Handles connection
errors gracefully in development mode (logs a warning, continues).
"""

import io
import logging
import uuid
from functools import lru_cache
from typing import Optional

from minio import Minio
from minio.error import S3Error
from urllib.parse import urlparse

from server.config import settings

logger = logging.getLogger(__name__)


def _get_minio_client() -> Minio:
    """Create a MinIO client from settings."""
    parsed = urlparse(settings.S3_ENDPOINT)
    # minio client wants host:port without scheme
    endpoint = parsed.netloc or parsed.path
    secure = settings.S3_USE_SSL

    return Minio(
        endpoint=endpoint,
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
        secure=secure,
        region=settings.S3_REGION,
    )


# Module-level client (reused across calls)
_client: Minio | None = None


def _get_client() -> Minio:
    """Lazy-init and return the singleton MinIO client."""
    global _client
    if _client is None:
        _client = _get_minio_client()
    return _client


def _ensure_bucket(client: Minio, bucket: str) -> None:
    """Create the bucket if it does not exist."""
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("Created MinIO bucket: %s", bucket)
    except Exception as exc:
        logger.warning("Could not ensure bucket '%s': %s", bucket, exc)


# ── Public async API ──────────────────────────────────────────────────────

async def upload_fundus_image(
    screening_id: str,
    eye: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
) -> str:
    """Upload a fundus image to MinIO and return the S3 key.

    Args:
        screening_id: UUID string of the screening.
        eye: 'left' or 'right'.
        image_bytes: Raw image bytes.
        content_type: MIME type (default image/jpeg).

    Returns:
        The S3 object key where the image was stored.
    """
    import asyncio

    bucket = settings.S3_BUCKET_IMAGES
    s3_key = f"screenings/{screening_id}/{eye}/{uuid.uuid4().hex}.jpg"

    def _upload():
        client = _get_client()
        _ensure_bucket(client, bucket)
        data = io.BytesIO(image_bytes)
        client.put_object(
            bucket_name=bucket,
            object_name=s3_key,
            data=data,
            length=len(image_bytes),
            content_type=content_type,
        )

    try:
        await asyncio.get_event_loop().run_in_executor(None, _upload)
        logger.info("Uploaded fundus image to %s/%s (%d bytes)", bucket, s3_key, len(image_bytes))
        return s3_key
    except Exception as exc:
        if settings.ENVIRONMENT.lower() in ("development", "dev", "local"):
            logger.warning(
                "MinIO upload failed (dev mode, continuing): %s", exc
            )
            return s3_key  # Return the key anyway so DB record is valid
        raise


async def download_fundus_image(s3_key: str) -> bytes:
    """Download a fundus image from MinIO by its S3 key.

    Args:
        s3_key: The object key in the images bucket.

    Returns:
        Raw image bytes.
    """
    import asyncio

    bucket = settings.S3_BUCKET_IMAGES

    def _download() -> bytes:
        client = _get_client()
        response = client.get_object(bucket_name=bucket, object_name=s3_key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        return data

    try:
        data = await asyncio.get_event_loop().run_in_executor(None, _download)
        logger.info("Downloaded %s/%s (%d bytes)", bucket, s3_key, len(data))
        return data
    except Exception as exc:
        if settings.ENVIRONMENT.lower() in ("development", "dev", "local"):
            logger.warning(
                "MinIO download failed (dev mode): %s", exc
            )
            raise
        raise


async def upload_report_pdf(
    screening_id: str,
    pdf_bytes: bytes,
) -> str:
    """Upload a generated PDF report to MinIO and return the S3 key.

    Args:
        screening_id: UUID string of the screening.
        pdf_bytes: Raw PDF bytes.

    Returns:
        The S3 object key where the PDF was stored.
    """
    import asyncio

    bucket = settings.S3_BUCKET_REPORTS
    s3_key = f"reports/{screening_id}/{uuid.uuid4().hex}.pdf"

    def _upload():
        client = _get_client()
        _ensure_bucket(client, bucket)
        data = io.BytesIO(pdf_bytes)
        client.put_object(
            bucket_name=bucket,
            object_name=s3_key,
            data=data,
            length=len(pdf_bytes),
            content_type="application/pdf",
        )

    try:
        await asyncio.get_event_loop().run_in_executor(None, _upload)
        logger.info("Uploaded PDF report to %s/%s (%d bytes)", bucket, s3_key, len(pdf_bytes))
        return s3_key
    except Exception as exc:
        if settings.ENVIRONMENT.lower() in ("development", "dev", "local"):
            logger.warning(
                "MinIO PDF upload failed (dev mode, continuing): %s", exc
            )
            return s3_key
        raise


# ── Legacy compatibility wrappers ─────────────────────────────────────────
# These preserve the old function signatures used by other parts of the
# codebase (e.g., the screening endpoint's original import of upload_image).

async def upload_image(
    image_bytes: bytes,
    bucket: str,
    key: str,
    content_type: str = "image/jpeg",
) -> str:
    """Legacy wrapper: Upload bytes to MinIO and return the key."""
    import asyncio

    def _upload():
        client = _get_client()
        _ensure_bucket(client, bucket)
        data = io.BytesIO(image_bytes)
        client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=data,
            length=len(image_bytes),
            content_type=content_type,
        )

    try:
        await asyncio.get_event_loop().run_in_executor(None, _upload)
        logger.info("Uploaded %s to %s/%s (%d bytes)", content_type, bucket, key, len(image_bytes))
        return key
    except Exception as exc:
        if settings.ENVIRONMENT.lower() in ("development", "dev", "local"):
            logger.warning("MinIO upload failed (dev mode): %s", exc)
            return key
        raise


async def download_image(bucket: str, key: str) -> bytes:
    """Legacy wrapper: Download an object from MinIO and return its bytes."""
    import asyncio

    def _download() -> bytes:
        client = _get_client()
        response = client.get_object(bucket_name=bucket, object_name=key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        return data

    try:
        data = await asyncio.get_event_loop().run_in_executor(None, _download)
        logger.info("Downloaded %s/%s (%d bytes)", bucket, key, len(data))
        return data
    except Exception as exc:
        if settings.ENVIRONMENT.lower() in ("development", "dev", "local"):
            logger.warning("MinIO download failed (dev mode): %s", exc)
            raise
        raise


async def generate_presigned_url(
    bucket: str,
    key: str,
    expiry: int | None = None,
) -> str:
    """Generate a presigned download URL."""
    import asyncio
    from datetime import timedelta

    if expiry is None:
        expiry = settings.S3_PRESIGNED_EXPIRY

    def _presign() -> str:
        client = _get_client()
        return client.presigned_get_object(
            bucket_name=bucket,
            object_name=key,
            expires=timedelta(seconds=expiry),
        )

    return await asyncio.get_event_loop().run_in_executor(None, _presign)


async def delete_object(bucket: str, key: str) -> None:
    """Delete an object from MinIO."""
    import asyncio

    def _delete():
        client = _get_client()
        client.remove_object(bucket_name=bucket, object_name=key)

    try:
        await asyncio.get_event_loop().run_in_executor(None, _delete)
        logger.info("Deleted %s/%s", bucket, key)
    except Exception as exc:
        logger.warning("Failed to delete %s/%s: %s", bucket, key, exc)


async def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    """List objects in a bucket with an optional prefix."""
    import asyncio

    def _list() -> list[dict]:
        client = _get_client()
        objects = client.list_objects(bucket_name=bucket, prefix=prefix)
        return [
            {
                "key": obj.object_name,
                "size": obj.size,
                "last_modified": str(obj.last_modified),
            }
            for obj in objects
        ]

    return await asyncio.get_event_loop().run_in_executor(None, _list)
