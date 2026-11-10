"""
emotion_model.py - DL model: what emotion does the child's face show?

Pipeline for one camera frame:
  base64 JPEG -> MediaPipe face detector -> crop the face
              -> MobileNetV2 CNN (PyTorch) -> probabilities over EMOTIONS

Needs weights/emotion_cnn.pt (train with ml_training/train_emotion_cnn.py).
If the weights, torch or MediaPipe are missing, predict_from_base64 returns
{"status": "model_unavailable"} instead of inventing numbers.

Camera frames are processed in memory and never stored.
"""
import base64
import binascii
import io
import logging
from pathlib import Path

import numpy as np

from .features import EMOTIONS

log = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "emotion_cnn.pt"
INPUT_SIZE = 128
FACE_MARGIN = 0.15      # extra context around the detected face box
MAX_SIDE_PX = 640       # larger frames are shrunk first to keep inference fast
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics,
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # same as in training

_state = {"loaded": False, "model": None, "detector": None, "torch": None}


def build_network(num_classes=len(EMOTIONS)):
    """The CNN architecture. Training and serving both call this so they match."""
    import torch
    import torchvision

    net = torchvision.models.mobilenet_v2(weights=None)
    net.classifier[1] = torch.nn.Linear(net.last_channel, num_classes)
    return net


def _load():
    if _state["loaded"]:
        return
    _state["loaded"] = True
    try:
        import mediapipe as mp
        import torch

        detector_cls = mp.solutions.face_detection.FaceDetection
    except (ImportError, AttributeError):
        log.warning(
            "torch or mediapipe (0.10.14 expected) unavailable; emotion analysis is off."
        )
        return
    if not WEIGHTS_PATH.exists():
        log.warning("No emotion model at %s; emotion analysis is off.", WEIGHTS_PATH)
        return
    model = build_network()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _state.update(
        model=model,
        torch=torch,
        detector=detector_cls(model_selection=0, min_detection_confidence=0.5),
    )


def predict_from_base64(image_b64):
    """
    Returns {"status": "ok", "emotion", "confidence", "probs"} or
    {"status": "model_unavailable" | "bad_image" | "no_face"}.
    """
    _load()
    if _state["model"] is None:
        return {"status": "model_unavailable"}

    try:
        from PIL import Image

        raw = base64.b64decode(image_b64, validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, ValueError, OSError):
        return {"status": "bad_image"}
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))

    detections = _state["detector"].process(np.asarray(img)).detections
    if not detections:
        return {"status": "no_face"}
    best = max(detections, key=lambda d: d.score[0])
    box = best.location_data.relative_bounding_box

    w, h = img.size
    x0 = max(0, int((box.xmin - FACE_MARGIN * box.width) * w))
    y0 = max(0, int((box.ymin - FACE_MARGIN * box.height) * h))
    x1 = min(w, int((box.xmin + box.width * (1 + FACE_MARGIN)) * w))
    y1 = min(h, int((box.ymin + box.height * (1 + FACE_MARGIN)) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"status": "no_face"}

    face = img.crop((x0, y0, x1, y1)).resize((INPUT_SIZE, INPUT_SIZE))
    arr = (np.asarray(face, dtype=np.float32) / 255.0 - _MEAN) / _STD
    torch = _state["torch"]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_state["model"](tensor), dim=1)[0].tolist()

    top = int(np.argmax(probs))
    return {
        "status": "ok",
        "emotion": EMOTIONS[top],
        "confidence": float(probs[top]),
        "probs": {name: float(p) for name, p in zip(EMOTIONS, probs)},
    }