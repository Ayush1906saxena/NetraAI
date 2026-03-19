"""Incoming webhook endpoints (Gupshup, ABDM callbacks, etc.)."""

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from server.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/gupshup")
async def gupshup_webhook(request: Request):
    """Handle incoming Gupshup delivery receipts and inbound messages."""
    try:
        body = await request.json()
    except Exception:
        body = dict(await request.form())

    event_type = body.get("type", "unknown")
    logger.info("Gupshup webhook received: type=%s", event_type)

    if event_type == "message-event":
        payload = body.get("payload", {})
        msg_status = payload.get("type", "")
        destination = payload.get("destination", "")
        logger.info("Message event: status=%s destination=%s", msg_status, destination)

    elif event_type == "message":
        payload = body.get("payload", {})
        sender = payload.get("sender", {}).get("phone", "")
        text = payload.get("payload", {}).get("text", "")
        logger.info("Inbound message from %s: %s", sender, text[:100])

    return {"status": "ok"}


@router.post("/abdm/callback")
async def abdm_callback(request: Request):
    """Handle ABDM gateway callbacks (consent, data push acknowledgements)."""
    body = await request.json()
    request_id = body.get("requestId", "")
    timestamp = body.get("timestamp", "")
    logger.info("ABDM callback: requestId=%s timestamp=%s", request_id, timestamp)

    # Dispatch based on endpoint hint
    resp = body.get("resp", {})
    if resp:
        logger.info("ABDM response status: %s", resp.get("statusCode"))

    return {"status": "acknowledged"}


@router.post("/payment")
async def payment_webhook(request: Request):
    """Placeholder for payment gateway webhooks."""
    body = await request.json()
    logger.info("Payment webhook received: %s", body.get("event", "unknown"))
    return {"status": "ok"}


@router.get("/test")
async def test_webhook():
    """Simple test endpoint to verify webhook routing."""
    return {"status": "ok", "message": "Webhook endpoint is reachable"}
