"""
hand_tracker.py — MediaPipe hand tracking that works on Python 3.12.

The legacy `mp.solutions` API is NOT available on 3.12, and LeRobot 0.5.1
*requires* 3.12 (so we can't downgrade). We use the modern Tasks API
(HandLandmarker) instead. The model file (~7 MB) auto-downloads on first use.

Both camera_test.py and follow.py use HandTracker so the MediaPipe details live
in exactly one place.
"""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Landmark index used as the tracked point (0 = wrist).
WRIST = 0

# 21-landmark hand skeleton (for drawing in camera_test.py).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("Downloading hand_landmarker.task (~7 MB, one time) ...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("Model downloaded.")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Could not download the hand model ({e}).\n"
                f"Download it manually from:\n  {_MODEL_URL}\n"
                f"and save it as:\n  {_MODEL_PATH}"
            )


class HandTracker:
    """Detects hands in a frame.

    `detect()` returns a LIST of hands (possibly empty). Each hand is a list of
    21 normalized (x, y) tuples; index WRIST (0) is the tracked point.

    We detect more than one hand on purpose: with a single-hand detector, the
    tracker flips between your commanding hand and the hand resting on your
    keyboard, and the arm jitters between them. follow.py picks the hand nearest
    the one it was already tracking.
    """

    def __init__(self, num_hands=2, min_confidence=0.5):
        _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,   # per-frame; no timestamp bookkeeping
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        return [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]

    def close(self):
        self._landmarker.close()
