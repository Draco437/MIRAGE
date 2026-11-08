"""
engagement_model.py - ML model: is the child engaged, distracted or stressed?

Uses a scikit-learn Random Forest saved at weights/engagement_rf.joblib
(train it with:  python -m ml_training.train_engagement_rf).

Until that file exists, a crude hand-written heuristic stands in so the whole
pipeline can run end to end. Every prediction says which one produced it in
its "source" field ("random_forest" or "heuristic"), so nobody mistakes the
stand-in for a trained model.
"""
import logging
from pathlib import Path

from .features import ENGAGEMENT_LABELS, FEATURE_NAMES, to_vector

log = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "engagement_rf.joblib"
JITTER_REFERENCE = 0.3  # hand jitter (m/s) treated as "maximum" by the heuristic

_bundle = None
_load_attempted = False


def _clip(x):
    return max(0.0, min(1.0, x))


def load_bundle(force=False):
    """Load the trained model once per process. Returns None if unavailable."""
    global _bundle, _load_attempted
    if _load_attempted and not force:
        return _bundle
    _load_attempted = True
    _bundle = None
    if not WEIGHTS_PATH.exists():
        log.warning("No trained engagement model at %s; using the heuristic stand-in.", WEIGHTS_PATH)
        return None
    try:
        import joblib

        bundle = joblib.load(WEIGHTS_PATH)
        if tuple(bundle["feature_names"]) != tuple(FEATURE_NAMES):
            log.error("Engagement model was trained on different features; retrain it. Using heuristic.")
            return None
        _bundle = bundle
    except Exception:  # corrupt file, sklearn version mismatch, ...
        log.exception("Could not load engagement model; using the heuristic stand-in.")
    return _bundle


def heuristic_predict(feats):
    """Crude stand-in: attention from eye/face looking, stress from distress and fidgeting."""
    attention = 0.6 * feats["face_rate"] + 0.4 * feats["eye_contact_rate"]
    stress = 0.0
    if feats["has_emotion"]:
        stress += 0.6 * feats["p_distressed"] + 0.1 * feats["p_sad"]
    stress += 0.2 * _clip(feats["hand_jitter_mean"] / JITTER_REFERENCE)
    if feats["has_response"]:
        stress += 0.1 * (1.0 - feats["response_accuracy"])
    stress = _clip(stress)

    scores = {
        "engaged": attention * (1.0 - stress),
        "distracted": 1.0 - attention,
        "stressed": stress,
    }
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probs": probs, "source": "heuristic"}


def predict(feats):
    """feats: dict from features.build_features. Returns label, confidence, probs, source."""
    bundle = load_bundle()
    if bundle is None:
        return heuristic_predict(feats)
    model = bundle["model"]
    proba = model.predict_proba([to_vector(feats)])[0]
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probs": probs, "source": "random_forest"}