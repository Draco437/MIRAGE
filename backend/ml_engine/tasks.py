"""
tasks.py - Celery tasks that run the ML pipeline, plus the small helpers
they share (saving results, pushing live updates).

Flow:
  analyze_frame   one camera frame -> emotion CNN -> save + show on dashboard
  analyze_window  last N seconds of data -> engagement model -> adaptation
                  rules -> if the scene should change, tell Unity

Both are triggered by core/consumers.py while a session is running.
Results are pushed over the Channels layer to two groups:
  unity_<session_id>      the VR client
  dashboard_<session_id>  the therapist's live view
"""
import logging
from datetime import datetime, timedelta, timezone

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.conf import settings

import db

from . import adaptation_rules, emotion_model, engagement_model
from .features import MIN_GAZE_SAMPLES, build_features

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


# --- Live push ----------------------------------------------------------
def _group_send(group, message):
    layer = get_channel_layer()
    try:
        async_to_sync(layer.group_send)(group, message)
    except Exception:  # a dead Redis must not crash the analysis
        log.exception("Could not push to %s", group)


def push_dashboard(session_id, kind, payload):
    """kind: emotion | engagement | adaptation | session_status"""
    _group_send(
        f"dashboard_{session_id}",
        {"type": "live.update", "payload": {"kind": kind, **payload}},
    )


def push_unity(session_id, payload):
    _group_send(f"unity_{session_id}", {"type": "adaptation.send", "payload": payload})


# --- Database helpers ---------------------------------------------------
def _session_is_active(session_id):
    oid = db.to_object_id(session_id)
    doc = db.col("sessions").find_one({"_id": oid}, {"status": 1}) if oid else None
    return bool(doc and doc.get("status") == "active")


def _latest_adaptation(session_id):
    return db.col("adaptation_events").find_one(
        {"session_id": session_id}, sort=[("received_at", -1)]
    )


def _save_and_push_adaptation(session_id, label, confidence, new_settings, reason, source):
    now = _now()
    db.col("adaptation_events").insert_one(
        {
            "session_id": session_id,
            "received_at": now,
            "source": source,  # "auto" or "therapist"
            "label": label,
            "confidence": confidence,
            "settings": new_settings,
            "reason": reason,
        }
    )
    payload = {
        "source": source,
        "label": label,
        "confidence": confidence,
        "settings": new_settings,
        "reason": reason,
    }
    push_unity(session_id, payload)
    push_dashboard(session_id, "adaptation", payload)
    return payload


def apply_therapist_override(session_id, partial_settings):
    """
    Called by the dashboard consumer. Merges the therapist's changes into the
    current settings and sends them to Unity. Returns the payload, or None if
    the session is not running.
    """
    if not _session_is_active(session_id):
        return None
    last = _latest_adaptation(session_id)
    current = last["settings"] if last else adaptation_rules.DEFAULT_SETTINGS
    merged = adaptation_rules.merge_settings(current, partial_settings)
    return _save_and_push_adaptation(
        session_id, None, None, merged, "set by therapist", "therapist"
    )


# --- Tasks --------------------------------------------------------------
@shared_task(name="ml_engine.analyze_frame", ignore_result=True)
def analyze_frame(session_id, image_b64):
    """Deep learning step: camera frame -> emotion."""
    result = emotion_model.predict_from_base64(image_b64)
    if result["status"] != "ok":
        # Tell the dashboard why there is no reading (e.g. face out of view).
        push_dashboard(session_id, "emotion", {"status": result["status"]})
        return result["status"]

    db.col("emotion_readings").insert_one(
        {
            "session_id": session_id,
            "received_at": _now(),
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "probs": result["probs"],
        }
    )
    push_dashboard(
        session_id,
        "emotion",
        {
            "status": "ok",
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "probs": result["probs"],
        },
    )
    return "ok"


@shared_task(name="ml_engine.analyze_window", ignore_result=True)
def analyze_window(session_id):
    """Machine learning step: recent behaviour -> engagement -> scene change."""
    if not _session_is_active(session_id):
        return "session_not_active"

    window = settings.MIRAGE["WINDOW_SECONDS"]
    now = _now()
    query = {"session_id": session_id, "received_at": {"$gte": now - timedelta(seconds=window)}}

    def fetch(name, fields):
        projection = {field: 1 for field in fields}
        projection["_id"] = 0
        return list(db.col(name).find(query, projection).sort("received_at", 1))

    gaze = fetch("gaze_events", ("target", "received_at"))
    if len(gaze) < MIN_GAZE_SAMPLES:
        return "insufficient_data"
    motor = fetch("motor_events", ("hand_speed", "jitter"))
    responses = fetch("response_events", ("response_time_ms", "correct"))
    emotions = fetch("emotion_readings", ("probs",))

    feats = build_features(gaze, motor, responses, emotions, window)
    pred = engagement_model.predict(feats)

    db.col("engagement_readings").insert_one(
        {
            "session_id": session_id,
            "received_at": now,
            "label": pred["label"],
            "confidence": pred["confidence"],
            "probs": pred["probs"],
            "source": pred["source"],
            "features": feats,
        }
    )
    push_dashboard(
        session_id,
        "engagement",
        {
            "label": pred["label"],
            "confidence": pred["confidence"],
            "probs": pred["probs"],
            "source": pred["source"],
            "features": feats,
        },
    )

    # Decide whether the scene should change.
    recent = [
        doc["label"]
        for doc in db.col("engagement_readings")
        .find({"session_id": session_id}, {"label": 1})
        .sort("received_at", -1)
        .limit(4)
    ][::-1]
    last = _latest_adaptation(session_id)
    if last:
        prev_settings = last["settings"]
        seconds_since = (now - last["received_at"]).total_seconds()
    else:
        prev_settings = adaptation_rules.DEFAULT_SETTINGS
        seconds_since = float("inf")

    decision = adaptation_rules.decide(
        pred["label"], pred["confidence"], recent, prev_settings, seconds_since
    )
    if decision.changed:
        _save_and_push_adaptation(
            session_id, pred["label"], pred["confidence"], decision.settings, decision.reason, "auto"
        )
        return "adapted"
    return "no_change"