"""
status.py - shows how far each dataset has got through the checklist, in order,
and tells you the next thing to do.

Run from backend/:   python -m ml_training.status

Gates 1-2 (access, licence) are manual; tick them off in DATASETS.md.
This script checks gates 3-6 by looking at your files:
  G3 placed    the dataset is in data/<folder>/
  G4 inspected inspect_report.txt exists and found that dataset
  G5 prepared  data/<folder>/_prepared/ has files (written by the prepare script)
  G6 trained   the trained model / reference file exists in ml_engine/weights/
"""
import argparse
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "ml_training" / "data"
WEIGHTS_DIR = BACKEND_DIR / "ml_engine" / "weights"

# (order, folder, name, what it becomes, file the training step must produce)
DATASETS = (
    (1, "hagrid", "HaGRID", "hand gesture -> responsiveness", "hand_gesture.joblib"),
    (2, "mmasd", "MMASD", "body movement -> arousal", "movement_reference.json"),
    (3, "saliency4asd", "Saliency4ASD", "gaze pattern -> attention", "gaze_reference.json"),
    (4, "liris_cse", "LIRIS-CSE", "face emotion -> distress", "emotion_cnn.pt"),
)

NEXT_STEP = {
    "placed": "download it and put the files in {folder} (read its licence first)",
    "inspected": "run:  python -m ml_training.inspect_datasets   and send me inspect_report.txt",
    "prepared": "run its prepare script (I will write it once I have seen your inspect report)",
    "trained": "run its training script, then wire it into the live pipeline",
}


def count_files(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "_prepared"]
        total += len(files)
    return total


def was_inspected(report, folder_name):
    """True if the inspect report has a section for this dataset that is not 'NOT FOUND'."""
    if not report.exists():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    marker = f"DATASET: {folder_name}"
    start = text.find(marker)
    if start < 0:
        return False
    end = text.find("DATASET: ", start + len(marker))
    section = text[start: end if end > 0 else len(text)]
    return "NOT FOUND" not in section


def gate_results(data_dir, weights_dir, folder, artifact):
    folder_path = data_dir / folder
    placed = folder_path.exists() and count_files(folder_path) > 0
    prepared_dir = folder_path / "_prepared"
    prepared = prepared_dir.exists() and any(prepared_dir.iterdir())
    return {
        "placed": placed,
        "inspected": placed and was_inspected(data_dir / "inspect_report.txt", folder),
        "prepared": prepared,
        "trained": (weights_dir / artifact).exists(),
    }


def mark(flag):
    return "yes" if flag else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    args = parser.parse_args()

    print("Dataset checklist (work through them in this order)\n")
    print(f"{'#':<2} {'Dataset':<13} {'Placed':<7} {'Inspected':<10} {'Prepared':<9} {'Trained':<8} Becomes")
    print("-" * 90)
    next_action = None
    for order, folder, name, becomes, artifact in DATASETS:
        g = gate_results(args.data_dir, args.weights_dir, folder, artifact)
        print(f"{order:<2} {name:<13} {mark(g['placed']):<7} {mark(g['inspected']):<10} "
              f"{mark(g['prepared']):<9} {mark(g['trained']):<8} {becomes}")
        if next_action is None:
            for gate in ("placed", "inspected", "prepared", "trained"):
                if not g[gate]:
                    next_action = (name, gate, args.data_dir / folder)
                    break

    print()
    if next_action is None:
        print("All four datasets are done. Next: wire the models together (ml_engine/fusion.py).")
    else:
        name, gate, folder_path = next_action
        print(f"NEXT: {name} - {NEXT_STEP[gate].format(folder=folder_path)}")
    print("\nReminder: gates 1-2 (access request, reading the licence) are manual. See DATASETS.md.")


if __name__ == "__main__":
    main()