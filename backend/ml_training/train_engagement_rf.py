"""
train_engagement_rf.py - trains the Random Forest that predicts engagement.

Reads a CSV with the columns in ml_engine.features.FEATURE_NAMES plus "label"
(engaged / distracted / stressed). If the CSV doesn't exist it generates the
synthetic one first (see the warning in generate_synthetic.py).

Run from backend/:   python -m ml_training.train_engagement_rf
Saves:               ml_engine/weights/engagement_rf.joblib
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.features import ENGAGEMENT_LABELS, FEATURE_NAMES  # noqa: E402
from ml_training.generate_synthetic import DEFAULT_OUT, generate  # noqa: E402

DEFAULT_MODEL_OUT = BACKEND_DIR / "ml_engine" / "weights" / "engagement_rf.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_OUT)
    args = parser.parse_args()

    if args.csv.exists():
        df = pd.read_csv(args.csv)
    else:
        print(f"{args.csv} not found - generating SYNTHETIC data (pipeline test only).")
        df = generate(1000)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)

    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing columns: {sorted(missing)}")
    unknown = set(df["label"]) - set(ENGAGEMENT_LABELS)
    if unknown:
        sys.exit(f"Unknown labels in CSV: {sorted(unknown)}")

    X = df[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"labels: {list(model.classes_)}")
    print(confusion_matrix(y_test, pred, labels=model.classes_))
    print("\nFeature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda p: -p[1]):
        print(f"  {name:24s} {imp:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": list(FEATURE_NAMES), "labels": list(model.classes_)},
        args.out,
    )
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()