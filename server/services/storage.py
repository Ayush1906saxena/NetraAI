"""MinIO / S3 storage operations."""

import logging
from typing import Optional

import aioboto3

from server.config import settings

logger = logging.getLogger(__name__)

_session = aioboto3.Session()


def _s3_config() -> dict:
    return {
        "endpoint_url": settings.S3_ENDPOINT,
        "aws_access_key_id": settings.S3_ACCESS_KEY,
        "aws_secret_access_key": settings.S3_SECRET_KEY,
        "region_name": settings.S3_REGION,
        "config": aioboto3.session.Config(signature_version="s3v4"),
    }


async def ensure_bucket(bucket: str) -> None:
    """Create the bucket if it does not exist."""
    async with _session.client("s3", **_s3_config()) as s3:
        try:
            await s3.head_bucket(Bucket=bucket)
        except Exception:
            try:
                await s3.create_bucket(Bucket=bucket)
                logger.info("Created bucket: %s", bucket)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", bucket, e)


async def upload_image(
    image_bytes: bytes,
    bucket: str,
    key: str,
    content_type: str = "image/jpeg",
) -> str:
    """Upload bytes to S3 and return the key."""
    await ensure_bucket(bucket)
    async with _session.client("s3", **_s3_config()) as s3:
        await s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )
    logger.info("Uploaded %s to %s/%s (%d bytes)", content_type, bucket, key, len(image_bytes))
    return key


async def download_image(bucket: str, key: str) -> bytes:
    """Download an object from S3 and return its bytes."""
    async with _session.client("s3", **_s3_config()) as s3:
        response = await s3.get_object(Bucket=bucket, Key=key)
        data = await response["Body"].read()
    logger.info("Downloaded %s/%s (%d bytes)", bucket, key, len(data))
    return data


async def generate_presigned_url(
    bucket: str,
    key: str,
    expiry: int | None = None,
) -> str:
    """Generate a presigned download URL."""
    if expiry is None:
        expiry = settings.S3_PRESIGNED_EXPIRY

    async with _session.client("s3", **_s3_config()) as s3:
        url = await s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiry,
        )
    return url


async def delete_object(bucket: str, key: str) -> None:
    """Delete an object from S3."""
    async with _session.client("s3", **_s3_config()) as s3:
        await s3.delete_object(Bucket=bucket, Key=key)
    logger.info("Deleted %s/%s", bucket, key)


async def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    """List objects in a bucket with an optional prefix."""
    async with _session.client("s3", **_s3_config()) as s3:
        response = await s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])
    return [{"key": obj["Key"], "size": obj["Size"], "last_modified": str(obj["LastModified"])} for obj in contents]
