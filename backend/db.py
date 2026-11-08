"""
db.py - MongoDB connection for Mirage.

All app data (children, sessions, sensor events, ML results) lives in MongoDB.
Django's own SQLite database only holds login accounts.

Every other module reaches the database through col("<name>"), so the list of
collections exists in exactly one place.
"""
import os
from functools import lru_cache

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING, MongoClient

# Collections that hold per-session data. Each document has a string
# "session_id" and a UTC "received_at" timestamp set by the server.
SESSION_DATA_COLLECTIONS = (
    "gaze_events",         # Unity: where the child is looking (target)
    "motor_events",        # Unity: hand speed and jitter
    "response_events",     # Unity: reaction to NPC prompts
    "session_events",      # Unity: scene_loaded, prompt_shown, ...
    "emotion_readings",    # ML (DL): emotion from a camera frame
    "engagement_readings", # ML: engaged / distracted / stressed
    "adaptation_events",   # scene changes sent to Unity (auto or therapist)
)

COLLECTIONS = ("children", "sessions") + SESSION_DATA_COLLECTIONS


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """One client per process. Created lazily so Celery's forked workers
    each build their own instead of sharing one across processes."""
    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)


def get_db():
    return get_client()[os.environ.get("MONGO_DB_NAME", "mirage")]


def col(name: str):
    if name not in COLLECTIONS:
        raise KeyError(f"Unknown collection: {name}")
    return get_db()[name]


def to_object_id(value):
    """Return an ObjectId, or None if value isn't a valid id."""
    try:
        return ObjectId(str(value))
    except (InvalidId, TypeError):
        return None


def ping() -> bool:
    get_client().admin.command("ping")
    return True


def ensure_indexes() -> None:
    """Idempotent. Makes the time-window queries and per-user lists fast."""
    for name in SESSION_DATA_COLLECTIONS:
        col(name).create_index([("session_id", ASCENDING), ("received_at", ASCENDING)])
    col("sessions").create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
    col("children").create_index("owner_id")