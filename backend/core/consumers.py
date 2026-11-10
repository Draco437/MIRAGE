"""
consumers.py - the live WebSocket endpoints (Django Channels).

UnityConsumer      ws/session/<id>/?token=<unity token>
    Receives sensor packets from the VR client, validates and stores them,
    and triggers the ML tasks. Sends scene changes back to Unity.

DashboardConsumer  ws/dashboard/<id>/?token=<jwt access token>
    Streams live results to the therapist and accepts manual overrides.

Both are async. PyMongo is blocking, so every database call goes through
sync_to_async to keep the event loop free for other connections.
"""
import json
import logging
import time
from datetime import datetime, timezone

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.conf import settings

import db
from ml_engine import tasks as ml_tasks
from ml_engine.adaptation_rules import DEFAULT_SETTINGS

from .serializers import PACKET_SERIALIZERS, SettingsSerializer

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _run_blocking(func, *args):
    return sync_to_async(func, thread_sensitive=False)(*args)


# --- Blocking database helpers (always called through _run_blocking) -----
def _load_session(session_id):
    oid = db.to_object_id(session_id)
    return db.col("sessions").find_one({"_id": oid}) if oid else None


def _insert(collection, doc):
    db.col(collection).insert_one(doc)


def _mark_active(session_id):
    db.col("sessions").update_one(
        {"_id": db.to_object_id(session_id)},
        {"$set": {"status": "active", "started_at": _now()}},
    )


def _finish_session(session_id):
    oid = db.to_object_id(session_id)
    doc = db.col("sessions").find_one({"_id": oid}, {"started_at": 1})
    if not doc:
        return
    ended = _now()
    started = doc.get("started_at")
    db.col("sessions").update_one(
        {"_id": oid},
        {
            "$set": {
                "status": "completed",
                "ended_at": ended,
                "duration_seconds": (ended - started).total_seconds() if started else None,
            }
        },
    )


def _error_detail(errors):
    """Turn DRF's error dict into plain JSON-safe text."""
    return {field: [str(e) for e in msgs] for field, msgs in errors.items()}


class _BaseConsumer(AsyncJsonWebsocketConsumer):
    """Shared plumbing: safe JSON parsing and error replies."""

    async def receive(self, text_data=None, bytes_data=None):
        if text_data is None:
            await self._error("text_only", "Binary messages are not supported.")
            return
        try:
            content = json.loads(text_data)
        except ValueError:
            await self._error("bad_json", "Message is not valid JSON.")
            return
        if not isinstance(content, dict):
            await self._error("bad_json", "Message must be a JSON object.")
            return
        await self.receive_json(content)

    async def _error(self, code, detail):
        await self.send_json({"type": "error", "code": code, "detail": detail})


class UnityConsumer(_BaseConsumer):
    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.group = f"unity_{self.session_id}"
        self.joined = False
        self.last_analysis = time.monotonic()
        self.last_frame = 0.0

        auth = self.scope.get("auth")
        if not auth or auth["kind"] != "unity" or auth["session_id"] != self.session_id:
            await self.close()  # not authenticated for this session
            return
        session = await _run_blocking(_load_session, self.session_id)
        if (
            session is None
            or session["owner_id"] != auth["user_id"]
            or session["status"] != "pending"
        ):
            await self.close()  # unknown, someone else's, or already used
            return

        await self.channel_layer.group_add(self.group, self.channel_name)
        self.joined = True
        await self.accept()
        await _run_blocking(_mark_active, self.session_id)
        await self.channel_layer.group_send(
            f"dashboard_{self.session_id}",
            {"type": "live.update", "payload": {"kind": "session_status", "status": "active"}},
        )
        # Tell Unity the starting scene settings.
        await self.send_json(
            {"type": "welcome", "session_id": self.session_id, "settings": DEFAULT_SETTINGS}
        )

    async def disconnect(self, close_code):
        if not getattr(self, "joined", False):
            return
        await self.channel_layer.group_discard(self.group, self.channel_name)
        await _run_blocking(_finish_session, self.session_id)
        await self.channel_layer.group_send(
            f"dashboard_{self.session_id}",
            {"type": "live.update", "payload": {"kind": "session_status", "status": "completed"}},
        )

    async def receive_json(self, content, **kwargs):
        packet_type = content.get("type")
        serializer_class = PACKET_SERIALIZERS.get(packet_type)
        if serializer_class is None:
            await self._error("unknown_type", f"Unknown packet type: {packet_type!r}")
            return
        serializer = serializer_class(data=content)
        if not serializer.is_valid():
            await self._error("invalid_packet", _error_detail(serializer.errors))
            return

        handler = {
            "gaze_data": self._on_gaze,
            "motor_data": self._on_motor,
            "response_event": self._on_response,
            "camera_frame": self._on_frame,
            "session_event": self._on_session_event,
        }[packet_type]
        await handler(serializer.validated_data)
        await self._maybe_analyze()

    # --- Packet handlers ---
    def _doc(self, **fields):
        return {"session_id": self.session_id, "received_at": _now(), **fields}

    async def _on_gaze(self, data):
        target = data["target"]
        await _run_blocking(
            _insert,
            "gaze_events",
            self._doc(
                client_t=data["t"],
                target=target,
                on_eyes=target == "eyes",
                on_face=target in ("eyes", "face"),
            ),
        )

    async def _on_motor(self, data):
        await _run_blocking(
            _insert,
            "motor_events",
            self._doc(client_t=data["t"], hand_speed=data["hand_speed"], jitter=data["jitter"]),
        )

    async def _on_response(self, data):
        await _run_blocking(
            _insert,
            "response_events",
            self._doc(
                client_t=data["t"],
                prompt_id=data["prompt_id"],
                response_time_ms=data["response_time_ms"],
                correct=data["correct"],
            ),
        )

    async def _on_session_event(self, data):
        await _run_blocking(
            _insert,
            "session_events",
            self._doc(client_t=data["t"], event=data["event"], detail=data["detail"]),
        )

    async def _on_frame(self, data):
        # Frames are not stored: they go straight to the emotion model.
        now = time.monotonic()
        if now - self.last_frame < settings.MIRAGE["FRAME_MIN_INTERVAL_SECONDS"]:
            return  # too soon after the last one; drop it silently
        self.last_frame = now
        try:
            await _run_blocking(ml_tasks.analyze_frame.delay, self.session_id, data["image"])
        except Exception:
            log.exception("Could not queue emotion analysis (is the Celery broker up?)")

    async def _maybe_analyze(self):
        """Every few seconds, ask the ML pipeline to look at the recent window."""
        now = time.monotonic()
        if now - self.last_analysis < settings.MIRAGE["ANALYSIS_INTERVAL_SECONDS"]:
            return
        self.last_analysis = now
        try:
            await _run_blocking(ml_tasks.analyze_window.delay, self.session_id)
        except Exception:
            log.exception("Could not queue engagement analysis (is the Celery broker up?)")

    # --- Messages pushed from the server side (ml_engine/tasks.py) ---
    async def adaptation_send(self, event):
        await self.send_json({"type": "adaptation", **event["payload"]})


class DashboardConsumer(_BaseConsumer):
    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.group = f"dashboard_{self.session_id}"
        self.joined = False

        auth = self.scope.get("auth")
        if not auth or auth["kind"] != "jwt":
            await self.close()
            return
        session = await _run_blocking(_load_session, self.session_id)
        if session is None or session["owner_id"] != auth["user_id"]:
            await self.close()
            return

        await self.channel_layer.group_add(self.group, self.channel_name)
        self.joined = True
        await self.accept()
        await self.send_json({"type": "connected", "session_id": self.session_id, "status": session["status"]})

    async def disconnect(self, close_code):
        if getattr(self, "joined", False):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def receive_json(self, content, **kwargs):
        if content.get("type") != "override":
            await self._error("unknown_type", "Only 'override' messages are accepted.")
            return
        serializer = SettingsSerializer(data=content.get("settings") or {})
        if not serializer.is_valid():
            await self._error("invalid_packet", _error_detail(serializer.errors))
            return
        if not serializer.validated_data:
            await self._error("invalid_packet", "No settings supplied.")
            return
        payload = await _run_blocking(
            ml_tasks.apply_therapist_override, self.session_id, dict(serializer.validated_data)
        )
        if payload is None:
            await self._error("session_not_active", "The VR session is not running.")

    # --- Messages pushed from the server side ---
    async def live_update(self, event):
        await self.send_json({"type": "live", **event["payload"]})