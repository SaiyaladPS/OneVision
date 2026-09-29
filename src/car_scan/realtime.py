"""Small in-process event hub used by the OneVision WebSocket endpoint."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

LOGGER = logging.getLogger(__name__)


@dataclass
class _Subscriber:
    websocket: WebSocket
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[dict[str, Any]]
    user_id: int
    session_id: str
    worker_snapshot_filter: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    camera_event_filter: Callable[[str], bool] | None = None


class OneVisionEventHub:
    """Fan out database events to all connected report clients.

    Database writes happen in synchronous worker threads while WebSocket
    clients are served by asyncio.  Each subscriber therefore owns an async
    queue and its event loop is woken with ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, _Subscriber] = {}

    async def connect(
        self,
        websocket: WebSocket,
        *,
        user_id: int,
        session_id: str,
        worker_snapshot_filter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        camera_event_filter: Callable[[str], bool] | None = None,
    ) -> _Subscriber:
        await websocket.accept()
        subscriber = _Subscriber(
            websocket=websocket,
            loop=asyncio.get_running_loop(),
            queue=asyncio.Queue(maxsize=100),
            user_id=int(user_id),
            session_id=str(session_id),
            worker_snapshot_filter=worker_snapshot_filter,
            camera_event_filter=camera_event_filter,
        )
        with self._lock:
            self._subscribers[uuid.uuid4().hex] = subscriber
        await websocket.send_json(
            {
                "id": f"connection-{uuid.uuid4().hex[:12]}",
                "type": "CONNECTED",
                "source": "OneVision",
                "message": "Connected to OneVision event stream",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "info",
            }
        )
        return subscriber

    def disconnect(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers = {
                key: item for key, item in self._subscribers.items() if item is not subscriber
            }

    @staticmethod
    def _enqueue(queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(payload)

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(self._enqueue, subscriber.queue, payload)
            except RuntimeError:
                self.disconnect(subscriber)

    @staticmethod
    async def _close_websocket(websocket: WebSocket, code: int) -> None:
        try:
            await websocket.close(code=code)
        except Exception:
            pass

    def close_user_sessions(self, user_id: int, *, keep_session_id: str | None = None) -> None:
        """Close WebSockets that no longer own the user's active session."""

        with self._lock:
            subscribers = [
                item
                for item in self._subscribers.values()
                if item.user_id == int(user_id) and (keep_session_id is None or item.session_id != keep_session_id)
            ]
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(
                    lambda item=subscriber: asyncio.create_task(self._close_websocket(item.websocket, 4001))
                )
            except RuntimeError:
                self.disconnect(subscriber)


event_hub = OneVisionEventHub()


def publish_event(
    event_type: str,
    *,
    scan_id: int | None = None,
    result: dict[str, Any] | None = None,
    plate: dict[str, Any] | None = None,
    message: str | None = None,
) -> None:
    """Publish a compact, safe event after a successful database write."""

    result = result or {}
    data: dict[str, Any] = {
        "scan_id": scan_id,
        "plate_count": int(result.get("plate_count") or 0),
        "media_type": str(result.get("media_type") or ""),
        "operator_name": str(result.get("operator_name") or ""),
    }
    if plate:
        data["plate"] = {
            "id": plate.get("id"),
            "country": plate.get("country"),
            "province": plate.get("province"),
            "plate_prefix": plate.get("plate_prefix"),
            "plate_number": plate.get("plate_number"),
            "confidence_level": plate.get("confidence_level"),
        }
    event_hub.publish(
        {
            "id": f"{event_type.lower()}-{scan_id or 'unknown'}-{uuid.uuid4().hex[:8]}",
            "type": event_type,
            "source": "OneVision",
            "message": message or "OneVision data was saved",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "success",
            "data": data,
        }
    )


def publish_worker_event(snapshot: dict[str, Any], *, message: str = "Worker updated") -> None:
    """Publish a worker snapshot without making the browser poll an API."""

    event_hub.publish(
        {
            "id": f"worker-{uuid.uuid4().hex[:12]}",
            "type": "WORKER_UPDATED",
            "source": "OneVision",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "info",
            "data": {"snapshot": snapshot},
        }
    )


def publish_roi_event(host: str, roi: dict[str, Any] | None, *, user: dict[str, Any]) -> None:
    """Broadcast an ROI edit so every authenticated display stays aligned."""

    event_hub.publish(
        {
            "id": f"roi-{host or 'default'}-{uuid.uuid4().hex[:8]}",
            "type": "ROI_UPDATED",
            "source": "OneVision",
            "message": "Camera ROI updated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "info",
            "data": {
                "host": str(host or ""),
                "roi": roi,
                "user": {
                    "id": int(user.get("id") or 0),
                    "display_name": str(user.get("display_name") or user.get("username") or ""),
                },
            },
        }
    )


async def serve_websocket(
    websocket: WebSocket,
    *,
    user_id: int,
    session_id: str,
    worker_snapshot_filter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    camera_event_filter: Callable[[str], bool] | None = None,
) -> None:
    subscriber = await event_hub.connect(
        websocket,
        user_id=user_id,
        session_id=session_id,
        worker_snapshot_filter=worker_snapshot_filter,
        camera_event_filter=camera_event_filter,
    )
    sender = asyncio.create_task(_send_events(subscriber))
    try:
        while True:
            # Keep the connection alive and allow the client to close cleanly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        event_hub.disconnect(subscriber)


async def _send_events(subscriber: _Subscriber) -> None:
    try:
        while True:
            event = await subscriber.queue.get()
            if event.get("type") == "ROI_UPDATED" and subscriber.camera_event_filter:
                host = str((event.get("data") or {}).get("host") or "")
                try:
                    if not subscriber.camera_event_filter(host):
                        continue
                except Exception:
                    LOGGER.exception("Unable to filter camera event for websocket user")
                    continue
            if event.get("type") == "WORKER_UPDATED" and subscriber.worker_snapshot_filter:
                try:
                    event = dict(event)
                    data = dict(event.get("data") or {})
                    snapshot = data.get("snapshot")
                    if isinstance(snapshot, dict):
                        data["snapshot"] = subscriber.worker_snapshot_filter(snapshot)
                    event["data"] = data
                except Exception:
                    LOGGER.exception("Unable to filter worker event for websocket user")
                    continue
            await subscriber.websocket.send_json(event)
    except (asyncio.CancelledError, WebSocketDisconnect):
        return
    except Exception:
        LOGGER.debug("OneVision WebSocket client disconnected", exc_info=True)
