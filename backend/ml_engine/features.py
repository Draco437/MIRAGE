"""
features.py - turns a window of raw session data into one feature row.

Shared by the live server (tasks.py) and offline training (ml_training/),
so the model always sees features computed the same way it was trained on.

This module is pure Python: no Django, no database. It takes plain lists of
dicts and returns plain numbers.
"""
from collections import defaultdict

# What Unity can report the child is looking at.
GAZE_TARGETS = ("eyes", "face", "body", "object", "away")

# Output classes of the emotion CNN. FER-style labels are merged into four:
# fear/anger/disgust -> distressed, surprise -> neutral.
EMOTIONS = ("happy", "neutral", "sad", "distressed")

# Output classes of the engagement model.
ENGAGEMENT_LABELS = ("engaged", "distracted", "stressed")

# Fixed order. The trained model depends on it, so only change it together
# with retraining.
FEATURE_NAMES = (
    "eye_contact_rate",     # share of gaze samples on the NPC's eyes (0-1)
    "face_rate",            # share on the eyes or face (0-1)
    "gaze_shift_rate",      # changes of gaze target per second
    "hand_speed_mean",      # mean hand speed (m/s)
    "hand_jitter_mean",     # mean hand-velocity std-dev (m/s)
    "response_time_ms_mean",
    "response_accuracy",    # share of prompts answered correctly (0-1)
    "p_happy",
    "p_neutral",
    "p_sad",
    "p_distressed",
    "has_emotion",          # 1 if any camera frame had a readable face
    "has_response",         # 1 if the child answered any prompt
)

# Values used when a data source had nothing in the window.
DEFAULT_RESPONSE_TIME_MS = 3000.0
DEFAULT_RESPONSE_ACCURACY = 0.5

# Fewer gaze samples than this in a window and the analysis is skipped.
MIN_GAZE_SAMPLES = 10


def _mean(values, default=0.0):
    return sum(values) / len(values) if values else default


def build_features(gaze, motor, responses, emotions, window_seconds):
    """
    gaze:       [{"target": "eyes", "received_at": datetime}, ...] oldest first
    motor:      [{"hand_speed": float, "jitter": float}, ...]
    responses:  [{"response_time_ms": float, "correct": bool}, ...]
    emotions:   [{"probs": {"happy": 0.1, ...}}, ...]
    Returns a dict with every name in FEATURE_NAMES.
    """
    n_gaze = len(gaze)
    eye_hits = sum(1 for g in gaze if g["target"] == "eyes")
    face_hits = sum(1 for g in gaze if g["target"] in ("eyes", "face"))
    shifts = sum(
        1 for prev, cur in zip(gaze, gaze[1:]) if prev["target"] != cur["target"]
    )

    feats = {
        "eye_contact_rate": eye_hits / n_gaze if n_gaze else 0.0,
        "face_rate": face_hits / n_gaze if n_gaze else 0.0,
        "gaze_shift_rate": shifts / float(window_seconds),
        "hand_speed_mean": _mean([m["hand_speed"] for m in motor]),
        "hand_jitter_mean": _mean([m["jitter"] for m in motor]),
        "response_time_ms_mean": _mean(
            [r["response_time_ms"] for r in responses], DEFAULT_RESPONSE_TIME_MS
        ),
        "response_accuracy": _mean(
            [1.0 if r["correct"] else 0.0 for r in responses], DEFAULT_RESPONSE_ACCURACY
        ),
        "has_response": 1.0 if responses else 0.0,
    }

    if emotions:
        sums = defaultdict(float)
        for e in emotions:
            for name in EMOTIONS:
                sums[name] += float(e["probs"].get(name, 0.0))
        for name in EMOTIONS:
            feats["p_" + name] = sums[name] / len(emotions)
        feats["has_emotion"] = 1.0
    else:
        for name in EMOTIONS:
            feats["p_" + name] = 1.0 / len(EMOTIONS)
        feats["has_emotion"] = 0.0

    return feats


def to_vector(feats):
    """Feature dict -> list in the order the model expects."""
    return [float(feats[name]) for name in FEATURE_NAMES]