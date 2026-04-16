"""
Emotion Detection Module
Uses MediaPipe Face Mesh to extract facial landmarks and heuristics,
with optional fallback to DeepFace for richer classification.

Exported:
    EmotionDetector — thread-safe, frame-rate-friendly detector
"""

import numpy as np
import cv2
from collections import deque
import threading

# Try optional heavy model (DeepFace); gracefully fall back to landmark heuristics
try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DEEPFACE_AVAILABLE = False

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ── Landmark indices relevant to expression ──────────────────────────────────
# Mouth corners
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
MOUTH_TOP   = 13
MOUTH_BOTTOM = 14
# Eyebrow inner
LEFT_BROW_INNER  = 107
RIGHT_BROW_INNER = 336
# Eye outer corners
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
# Eye heights
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM = 374

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised"]


def _heuristic_emotion(landmarks) -> str:
    """Compute emotion from geometric ratios."""
    pts = {i: np.array([landmarks[i].x, landmarks[i].y])
           for i in [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM,
                     LEFT_BROW_INNER, RIGHT_BROW_INNER,
                     LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
                     LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                     RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]}

    face_width = np.linalg.norm(pts[LEFT_EYE_OUTER] - pts[RIGHT_EYE_OUTER])
    if face_width < 1e-5:
        return "neutral"

    # Mouth curve: y of corners vs midpoint (positive = smile)
    mid_mouth_y = (pts[MOUTH_LEFT][1] + pts[MOUTH_RIGHT][1]) / 2
    smile_ratio = (pts[MOUTH_TOP][1] - mid_mouth_y) / face_width

    # Mouth openness
    mouth_open = (pts[MOUTH_BOTTOM][1] - pts[MOUTH_TOP][1]) / face_width

    # Brow height relative to eye corners (lower = angry/concerned)
    avg_brow_y  = (pts[LEFT_BROW_INNER][1] + pts[RIGHT_BROW_INNER][1]) / 2
    avg_eye_y   = (pts[LEFT_EYE_OUTER][1]  + pts[RIGHT_EYE_OUTER][1])  / 2
    brow_raise  = (avg_eye_y - avg_brow_y) / face_width

    # Eye openness
    left_eye_h  = abs(pts[LEFT_EYE_TOP][1]  - pts[LEFT_EYE_BOTTOM][1])
    right_eye_h = abs(pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYE_BOTTOM][1])
    eye_open    = ((left_eye_h + right_eye_h) / 2) / face_width

    # Decision tree
    if mouth_open > 0.09 and eye_open > 0.07:
        return "surprised"
    if smile_ratio > 0.02 and brow_raise > 0.12:
        return "happy"
    if brow_raise < 0.08:
        return "angry"
    if smile_ratio < -0.01 and brow_raise < 0.12:
        return "sad"
    return "neutral"


class EmotionDetector:
    """Thread-safe, low-overhead emotion detector.

    Usage:
        detector = EmotionDetector()
        # In your frame loop:
        emotion = detector.update(bgr_frame)
    """

    def __init__(self, smoothing: int = 10, use_deepface: bool = False):
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._history: deque = deque(maxlen=smoothing)
        self._current = "neutral"
        self._lock = threading.Lock()
        self._use_deepface = use_deepface and _DEEPFACE_AVAILABLE
        self._frame_counter = 0
        self._deepface_interval = 15   # run DeepFace every N frames (slow)

        if self._use_deepface:
            print("[EmotionDetector] Using DeepFace backend.")
        else:
            print("[EmotionDetector] Using MediaPipe heuristic backend.")

    def update(self, bgr_frame: np.ndarray) -> str:
        """Process a BGR frame and return smoothed emotion string."""
        self._frame_counter += 1
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        raw_emotion = "neutral"

        if result.multi_face_landmarks:
            face_lm = result.multi_face_landmarks[0].landmark

            if self._use_deepface and self._frame_counter % self._deepface_interval == 0:
                try:
                    analysis = DeepFace.analyze(
                        bgr_frame, actions=["emotion"],
                        enforce_detection=False, silent=True
                    )
                    raw_emotion = analysis[0]["dominant_emotion"]
                    # Normalise to our label set
                    mapping = {
                        "fear": "surprised", "disgust": "angry",
                        "contempt": "angry"
                    }
                    raw_emotion = mapping.get(raw_emotion, raw_emotion)
                    if raw_emotion not in EMOTIONS:
                        raw_emotion = "neutral"
                except Exception:
                    raw_emotion = _heuristic_emotion(face_lm)
            else:
                raw_emotion = _heuristic_emotion(face_lm)
        else:
            raw_emotion = "neutral"

        with self._lock:
            self._history.append(raw_emotion)
            # Majority vote for smooth output
            self._current = max(set(self._history),
                                key=self._history.count)

        return self._current

    @property
    def emotion(self) -> str:
        with self._lock:
            return self._current

    def close(self):
        self._face_mesh.close()
