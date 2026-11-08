"""
generate_synthetic.py - makes fake engagement data so the pipeline can be
tested before any real sessions exist.

IMPORTANT: this data follows assumptions written in this file, not real
children's behaviour. A model trained on it will score almost perfectly
because the classes are separable by construction. That proves the code
works; it says nothing about real-world accuracy. Replace it with
therapist-labelled session data before drawing any conclusions.

Run from backend/:   python -m ml_training.generate_synthetic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.features import EMOTIONS, FEATURE_NAMES  # noqa: E402

DEFAULT_OUT = BACKEND_DIR / "ml_training" / "data" / "synthetic_engagement.csv"

# Per class: (mean, std) of each numeric feature, and Dirichlet weights for
# emotion probabilities in the order of EMOTIONS (happy, neutral, sad, distressed).
PROFILES = {
    "engaged": {
        "eye_contact_rate": (0.45, 0.15), "face_rate": (0.80, 0.10), "gaze_shift_rate": (0.15, 0.08),
        "hand_speed_mean": (0.30, 0.10), "hand_jitter_mean": (0.05, 0.03),
        "response_time_ms_mean": (2000, 600), "response_accuracy": (0.80, 0.12),
        "emotion_alpha": (4.0, 4.0, 1.0, 0.5),
    },
    "distracted": {
        "eye_contact_rate": (0.08, 0.06), "face_rate": (0.30, 0.15), "gaze_shift_rate": (0.50, 0.20),
        "hand_speed_mean": (0.20, 0.10), "hand_jitter_mean": (0.08, 0.04),
        "response_time_ms_mean": (4500, 1200), "response_accuracy": (0.50, 0.20),
        "emotion_alpha": (1.5, 4.0, 1.5, 1.0),
    },
    "stressed": {
        "eye_contact_rate": (0.05, 0.05), "face_rate": (0.40, 0.20), "gaze_shift_rate": (0.70, 0.25),
        "hand_speed_mean": (0.60, 0.25), "hand_jitter_mean": (0.25, 0.10),
        "response_time_ms_mean": (3500, 1500), "response_accuracy": (0.45, 0.20),
        "emotion_alpha": (0.5, 1.5, 2.0, 5.0),
    },
}
RATE_COLUMNS = ("eye_contact_rate", "face_rate", "response_accuracy")


def generate(n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for label, profile in PROFILES.items():
        for _ in range(n_per_class):
            row = {}
            for name in FEATURE_NAMES:
                if name in profile:
                    mean, std = profile[name]
                    value = rng.normal(mean, std)
                    row[name] = float(np.clip(value, 0, 1)) if name in RATE_COLUMNS else float(max(value, 0))
            row["has_emotion"] = float(rng.random() < 0.9)
            if row["has_emotion"]:
                probs = rng.dirichlet(profile["emotion_alpha"])
            else:
                probs = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
            for emo, p in zip(EMOTIONS, probs):
                row["p_" + emo] = float(p)
            row["has_response"] = float(rng.random() < 0.6)
            if not row["has_response"]:
                row["response_time_ms_mean"] = 3000.0
                row["response_accuracy"] = 0.5
            row["label"] = label
            rows.append(row)
    return pd.DataFrame(rows)[list(FEATURE_NAMES) + ["label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="rows per class")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    df = generate(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()