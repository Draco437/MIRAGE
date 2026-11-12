"""
signals.py - the common shape every sensor model reports its result in.

Each model (face emotion, hand gesture, body movement, gaze pattern) looks at a
different kind of data, but each hands the rest of the system the same thing:
a Signal. That lets fusion.py combine them without knowing how any of them work.

All values are 0..1 and, where it makes sense, relative to THIS child's own
baseline (not to a "typical child"), because children with autism can differ a
lot from population averages.

Pure Python: no Django, no database.
"""
from dataclasses import dataclass

# What each signal means:
#   attention       how much the child is attending to the character (gaze / face direction)
#   distress        how strongly the face shows distress (emotion model)
#   arousal         how agitated the body movement is compared with the child's own baseline
#   responsiveness  how much the child is responding (gestures, answers)
SIGNAL_NAMES = ("attention", "distress", "arousal", "responsiveness")


@dataclass(frozen=True)
class Signal:
    name: str          # one of SIGNAL_NAMES
    value: float       # 0..1
    confidence: float  # 0..1, how much to trust this reading (image quality, model certainty, ...)
    source: str        # which model produced it, e.g. "face_emotion"

    def __post_init__(self):
        if self.name not in SIGNAL_NAMES:
            raise ValueError(f"Unknown signal name: {self.name!r}")
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Signal value must be between 0 and 1, got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Signal confidence must be between 0 and 1, got {self.confidence}")