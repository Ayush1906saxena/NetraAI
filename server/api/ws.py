"""
WebSocket endpoint for real-time screening status updates.

Clients connect to /v1/ws/screening/{screening_id} and receive JSON
status updates as the screening progresses through its lifecycle:
  - {"status": "uploading"}
  - {"status": "analyzing"}
  - {"status": "complete", "results": {...}}
  - {"status": "error", "detail": "..."}
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory connection manager ──────────────────────────────────────────

_active_connections: dict[str, list[WebSocket]] = {}


async def _add_connection(screening_id: str, ws: WebSocket) -> None:
    """Register a WebSocket connection for a screening."""
    if screening_id not in _active_connections:
        _active_connections[screening_id] = []
    _active_connections[screening_id].append(ws)
    logger.info(
        "WebSocket connected for screening %s (total: %d)",
        screening_id,
        len(_active_connections[screening_id]),
    )


async def _remove_connection(screening_id: str, ws: WebSocket) -> None:
    """Unregister a WebSocket connection."""
    if screening_id in _active_connections:
        try:
            _active_connections[screening_id].remove(ws)
        except ValueError:
            pass
        if not _active_connections[screening_id]:
            del _active_connections[screening_id]
    logger.info("WebSocket disconnected for screening %s", screening_id)


async def broadcast_screening_update(
    screening_id: str,
    data: dict[str, Any],
) -> None:
    """Send a JSON update to all connected clients for a screening.

    This function is safe to call even when no clients are connected
    (it simply does nothing). Failed sends are logged and the dead
    connection is removed.
    """
    connections = _active_connections.get(screening_id, [])
    if not connections:
        return

    message = json.dumps(data)
    dead: list[WebSocket] = []

    for ws in connections:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(message)
        except Exception as exc:
            logger.warning(
                "Failed to send WS update for screening %s: %s",
                screening_id,
                exc,
            )
            dead.append(ws)

    # Clean up dead connections
    for ws in dead:
        await _remove_connection(screening_id, ws)


def get_connected_screenings() -> list[str]:
    """Return screening IDs that have active WebSocket connections."""
    return list(_active_connections.keys())


# ── WebSocket endpoint ────────────────────────────────────────────────────

@router.websocket("/screening/{screening_id}")
async def screening_ws(
    websocket: WebSocket,
    screening_id: uuid.UUID,
):
    """WebSocket for real-time screening progress updates.

    Clients should connect here before triggering analysis. They will
    receive JSON messages as the screening status changes. The connection
    stays open until the client disconnects or the screening reaches a
    terminal state (complete / error).

    The client can also send JSON messages (e.g., ping / keep-alive);
    these are logged but not acted upon.
    """
    await websocket.accept()
    sid = str(screening_id)
    await _add_connection(sid, websocket)

    # Send initial connected acknowledgment
    try:
        await websocket.send_json({
            "status": "connected",
            "screening_id": sid,
            "message": "Listening for screening updates.",
        })
    except Exception:
        await _remove_connection(sid, websocket)
        return

    try:
        while True:
            # Keep connection alive; read any client messages
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=300,  # 5 min idle timeout
                )
                logger.debug("WS message from client for screening %s: %s", sid, raw[:200])
            except asyncio.TimeoutError:
                # Send a ping-like keep-alive
                try:
                    await websocket.send_json({"status": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("Client disconnected from screening WS %s", sid)
    except Exception as exc:
        logger.warning("WebSocket error for screening %s: %s", sid, exc)
    finally:
        await _remove_connection(sid, websocket)
