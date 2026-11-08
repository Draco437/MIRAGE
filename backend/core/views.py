"""
views.py - the REST API used by the React dashboard.

Every endpoint except health, register and login requires a JWT, and every
child/session is checked against the logged-in user, so a therapist can only
see their own data. Data lives in MongoDB (see db.py).
"""
import re
from collections import Counter
from datetime import datetime, timezone

from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

import db
from ml_engine.features import EMOTIONS, ENGAGEMENT_LABELS

from .auth import make_unity_token
from .serializers import (
    OBJECT_ID,
    ChildSerializer,
    RegisterSerializer,
    SessionCreateSerializer,
)

_OBJECT_ID_RE = re.compile(OBJECT_ID)
TIMELINE_LIMIT = 5000


# --- Small helpers ------------------------------------------------------
def _now():
    return datetime.now(timezone.utc)


def _oid_or_404(value):
    oid = db.to_object_id(value) if _OBJECT_ID_RE.match(str(value)) else None
    if oid is None:
        raise NotFound()
    return oid


def _owned_child(request, child_id):
    doc = db.col("children").find_one({"_id": _oid_or_404(child_id), "owner_id": request.user.id})
    if doc is None:
        raise NotFound("Child not found.")
    return doc


def _owned_session(request, session_id):
    doc = db.col("sessions").find_one({"_id": _oid_or_404(session_id), "owner_id": request.user.id})
    if doc is None:
        raise NotFound("Session not found.")
    return doc


def _child_out(doc):
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "age": doc["age"],
        "notes": doc.get("notes", ""),
        "created_at": doc["created_at"],
    }


def _session_out(doc):
    return {
        "id": str(doc["_id"]),
        "child_id": doc["child_id"],
        "scenario": doc["scenario"],
        "status": doc["status"],
        "created_at": doc["created_at"],
        "started_at": doc.get("started_at"),
        "ended_at": doc.get("ended_at"),
        "duration_seconds": doc.get("duration_seconds"),
    }


def _purge_sessions(session_ids):
    """Permanently delete sessions and every event recorded for them."""
    if not session_ids:
        return
    for name in db.SESSION_DATA_COLLECTIONS:
        db.col(name).delete_many({"session_id": {"$in": session_ids}})
    db.col("sessions").delete_many({"_id": {"$in": [db.to_object_id(s) for s in session_ids]}})


# --- Per-session statistics (shared by analytics and progress) -----------
def _gaze_stats(session_id):
    pipeline = [
        {"$match": {"session_id": session_id}},
        {
            "$group": {
                "_id": None,
                "n": {"$sum": 1},
                "eyes": {"$sum": {"$cond": ["$on_eyes", 1, 0]}},
                "face": {"$sum": {"$cond": ["$on_face", 1, 0]}},
            }
        },
    ]
    rows = list(db.col("gaze_events").aggregate(pipeline))
    if not rows:
        return {"samples": 0, "eye_contact_rate": None, "face_rate": None}
    r = rows[0]
    return {
        "samples": r["n"],
        "eye_contact_rate": r["eyes"] / r["n"],
        "face_rate": r["face"] / r["n"],
    }


def _response_stats(session_id):
    pipeline = [
        {"$match": {"session_id": session_id}},
        {
            "$group": {
                "_id": None,
                "n": {"$sum": 1},
                "correct": {"$sum": {"$cond": ["$correct", 1, 0]}},
                "mean_ms": {"$avg": "$response_time_ms"},
            }
        },
    ]
    rows = list(db.col("response_events").aggregate(pipeline))
    if not rows:
        return {"prompts": 0, "accuracy": None, "mean_response_time_ms": None}
    r = rows[0]
    return {"prompts": r["n"], "accuracy": r["correct"] / r["n"], "mean_response_time_ms": r["mean_ms"]}


def _engagement_stats(session_id, with_timeline=False):
    docs = list(
        db.col("engagement_readings")
        .find({"session_id": session_id}, {"_id": 0, "received_at": 1, "label": 1, "confidence": 1, "source": 1})
        .sort("received_at", 1)
        .limit(TIMELINE_LIMIT)
    )
    counts = Counter(d["label"] for d in docs)
    total = len(docs)
    out = {
        "readings": total,
        "counts": {label: counts.get(label, 0) for label in ENGAGEMENT_LABELS},
        "ratios": {label: (counts.get(label, 0) / total if total else None) for label in ENGAGEMENT_LABELS},
        "sources": sorted({d["source"] for d in docs}),
    }
    if with_timeline:
        out["timeline"] = [
            {"t": d["received_at"], "label": d["label"], "confidence": d["confidence"]} for d in docs
        ]
    return out


def _emotion_stats(session_id, with_timeline=False):
    docs = list(
        db.col("emotion_readings")
        .find({"session_id": session_id}, {"_id": 0, "received_at": 1, "emotion": 1, "confidence": 1, "probs": 1})
        .sort("received_at", 1)
        .limit(TIMELINE_LIMIT)
    )
    counts = Counter(d["emotion"] for d in docs)
    total = len(docs)
    out = {
        "readings": total,
        "counts": {e: counts.get(e, 0) for e in EMOTIONS},
        "mean_probs": {e: (sum(d["probs"].get(e, 0.0) for d in docs) / total if total else None) for e in EMOTIONS},
    }
    if with_timeline:
        out["timeline"] = [
            {"t": d["received_at"], "emotion": d["emotion"], "confidence": d["confidence"]} for d in docs
        ]
    return out


def _adaptation_list(session_id):
    docs = (
        db.col("adaptation_events")
        .find({"session_id": session_id}, {"_id": 0, "session_id": 0})
        .sort("received_at", 1)
        .limit(TIMELINE_LIMIT)
    )
    return [{"t": d["received_at"], **{k: v for k, v in d.items() if k != "received_at"}} for d in docs]


# --- Endpoints ----------------------------------------------------------
class HealthView(APIView):
    """GET /api/health/ - is the API up and is MongoDB reachable?"""

    permission_classes = [AllowAny]
    throttle_classes = []

    def get(self, request):
        try:
            db.ping()
        except Exception:
            return Response({"status": "degraded", "mongo": False}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({"status": "ok", "mongo": True})


class RegisterView(APIView):
    """POST /api/auth/register/ - create a therapist/parent account."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({"id": user.id, "username": user.username}, status=status.HTTP_201_CREATED)


class ChildListCreateView(APIView):
    """GET /api/children/ - list your children. POST - add one."""

    def get(self, request):
        docs = db.col("children").find({"owner_id": request.user.id}).sort("created_at", -1)
        return Response([_child_out(d) for d in docs])

    def post(self, request):
        serializer = ChildSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        doc = {**serializer.validated_data, "owner_id": request.user.id, "created_at": _now()}
        result = db.col("children").insert_one(doc)
        doc["_id"] = result.inserted_id
        return Response(_child_out(doc), status=status.HTTP_201_CREATED)


class ChildDetailView(APIView):
    """GET /api/children/<id>/. DELETE removes the child and ALL their session data."""

    def get(self, request, child_id):
        return Response(_child_out(_owned_child(request, child_id)))

    def delete(self, request, child_id):
        child = _owned_child(request, child_id)
        sessions = list(db.col("sessions").find({"child_id": str(child["_id"])}, {"status": 1}))
        if any(s["status"] == "active" for s in sessions):
            return Response({"detail": "A session is running; end it first."}, status=status.HTTP_409_CONFLICT)
        _purge_sessions([str(s["_id"]) for s in sessions])
        db.col("children").delete_one({"_id": child["_id"]})
        return Response(status=status.HTTP_204_NO_CONTENT)


class ChildProgressView(APIView):
    """GET /api/children/<id>/progress/ - one summary row per completed session."""

    def get(self, request, child_id):
        child = _owned_child(request, child_id)
        sessions = db.col("sessions").find(
            {"child_id": str(child["_id"]), "status": "completed"}
        ).sort("created_at", 1)
        rows = []
        for s in sessions:
            sid = str(s["_id"])
            engagement = _engagement_stats(sid)
            rows.append(
                {
                    "session_id": sid,
                    "scenario": s["scenario"],
                    "started_at": s.get("started_at"),
                    "duration_seconds": s.get("duration_seconds"),
                    "eye_contact_rate": _gaze_stats(sid)["eye_contact_rate"],
                    "engagement_ratios": engagement["ratios"],
                    "engagement_readings": engagement["readings"],
                }
            )
        return Response({"child": _child_out(child), "sessions": rows})


class SessionListCreateView(APIView):
    """GET /api/sessions/?child_id= - your sessions. POST - create one for a child."""

    def get(self, request):
        query = {"owner_id": request.user.id}
        child_id = request.query_params.get("child_id")
        if child_id:
            if not _OBJECT_ID_RE.match(child_id):
                raise NotFound()
            query["child_id"] = child_id
        docs = db.col("sessions").find(query).sort("created_at", -1).limit(100)
        return Response([_session_out(d) for d in docs])

    def post(self, request):
        serializer = SessionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        child = _owned_child(request, serializer.validated_data["child_id"])

        doc = {
            "owner_id": request.user.id,
            "child_id": str(child["_id"]),
            "scenario": serializer.validated_data["scenario"],
            "status": "pending",
            "created_at": _now(),
            "started_at": None,
            "ended_at": None,
            "duration_seconds": None,
        }
        session_id = str(db.col("sessions").insert_one(doc).inserted_id)

        scheme = "wss" if request.is_secure() else "ws"
        host = request.get_host()
        token = make_unity_token(session_id, request.user.id)
        return Response(
            {
                "session_id": session_id,
                # Give these to Unity: the token is valid for this session only.
                "unity_ws_url": f"{scheme}://{host}/ws/session/{session_id}/?token={token}",
                # The dashboard appends its own login token: ?token=<access token>
                "dashboard_ws_url": f"{scheme}://{host}/ws/dashboard/{session_id}/",
            },
            status=status.HTTP_201_CREATED,
        )


class SessionDetailView(APIView):
    """GET /api/sessions/<id>/. DELETE removes the session and its data."""

    def get(self, request, session_id):
        return Response(_session_out(_owned_session(request, session_id)))

    def delete(self, request, session_id):
        session = _owned_session(request, session_id)
        if session["status"] == "active":
            return Response({"detail": "Session is running; end it first."}, status=status.HTTP_409_CONFLICT)
        _purge_sessions([str(session["_id"])])
        return Response(status=status.HTTP_204_NO_CONTENT)


class SessionAnalyticsView(APIView):
    """GET /api/sessions/<id>/analytics/ - everything the session page needs."""

    def get(self, request, session_id):
        session = _owned_session(request, session_id)
        sid = str(session["_id"])
        return Response(
            {
                "session": _session_out(session),
                "gaze": _gaze_stats(sid),
                "responses": _response_stats(sid),
                "engagement": _engagement_stats(sid, with_timeline=True),
                "emotion": _emotion_stats(sid, with_timeline=True),
                "adaptations": _adaptation_list(sid),
            }
        )