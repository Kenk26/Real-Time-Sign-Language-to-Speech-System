"""
Step 1: Data Collection
Captures webcam video, detects hand landmarks using MediaPipe Tasks API,
stores sequences of 30 frames as .npy files.

Uses the same MediaPipe Tasks API as gesture_controller.py
(mediapipe.tasks.python.vision.HandLandmarker) instead of the legacy
mp.solutions.hands API.

Usage:
    python 1_collect_data.py --word "hello" --sequences 30
    python 1_collect_data.py --word "thanks" --sequences 30
    python 1_collect_data.py --word "yes" --sequences 30
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import argparse
import time
import urllib.request

# ── MediaPipe model setup ────────────────────────────────────────────────────
MEDIAPIPE_MODEL_PATH = "hand_landmarker.task"
MEDIAPIPE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

SEQUENCE_LENGTH = 30      # frames per gesture sequence
DATASET_DIR     = "dataset"

# Skeleton connections for drawing (matches gesture_controller.py)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17)
]


def download_model():
    """Download the MediaPipe hand landmarker model if not present."""
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print("Downloading hand landmarker model (~13 MB)...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
        print("  Model downloaded.")


def create_detector():
    """Create and return a MediaPipe HandLandmarker (Tasks API)."""
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(result) -> np.ndarray:
    """
    Extracts landmarks for both hands (Left and Right).
    Returns a flattened 126-dim vector (63 for Left, 63 for Right).
    Missing hands are padded with zeros. Includes a safeguard for MediaPipe glitches.
    """
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    if result.hand_landmarks:
        for idx, hand_info in enumerate(result.handedness):
            label = hand_info[0].category_name  # Usually "Left" or "Right"
            lms = result.hand_landmarks[idx]
            
            # Normalize relative to wrist (landmark 0)
            base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
            coords = []
            for lm in lms:
                coords.append(lm.x - base_x)
                coords.append(lm.y - base_y)
                coords.append(lm.z - base_z)
            
            coords_array = np.array(coords, dtype=np.float32)
            
            # Safeguard: if label is "Left" but lh is already filled, put it in rh
            if label == "Left":
                if np.all(lh == 0): # If lh is empty
                    lh = coords_array
                else:               # MediaPipe glitched and found two Lefts
                    rh = coords_array
            else:
                if np.all(rh == 0): # If rh is empty
                    rh = coords_array
                else:               # MediaPipe glitched and found two Rights
                    lh = coords_array

    return np.concatenate([lh, rh])


def draw_skeleton(frame, lms, color=(0, 255, 180)):
    """Draw hand landmarks and connections onto frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, color, -1)


def collect_word(word: str, num_sequences: int = 30):
    download_model()

    save_dir = os.path.join(DATASET_DIR, word)
    os.makedirs(save_dir, exist_ok=True)

    # Find next sequence index
    existing  = [int(f) for f in os.listdir(save_dir) if f.isdigit()]
    start_idx = max(existing) + 1 if existing else 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    detector = create_detector()

    print(f"\n[INFO] Collecting '{word}' — {num_sequences} sequences "
          f"(starting at index {start_idx})")
    print("[INFO] Press SPACE to start each sequence, Q to quit.\n")

    seq_idx   = start_idx
    collected = 0

    try:
        while collected < num_sequences:
            # ── Countdown / ready screen ─────────────────────────────────
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Run detector so we can show live skeleton while waiting
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)
                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        draw_skeleton(frame, hand_lms)

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80),
                              (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame,
                            f"Word: '{word}' | Seq {collected+1}/{num_sequences}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 180), 2)
                cv2.putText(frame, "Press SPACE to record",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 1)
                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    waiting = False
                elif key == ord("q"):
                    print("[INFO] Quit.")
                    return

            # ── Record sequence ──────────────────────────────────────────
            seq_save_dir = os.path.join(save_dir, str(seq_idx))
            os.makedirs(seq_save_dir, exist_ok=True)

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Detect with Tasks API
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                # Draw skeleton
                # Inside the recording loops (waiting and recording):
                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        draw_skeleton(frame, hand_lms)

                landmarks = extract_hand_landmarks(result)

                # Save individual frame
                np.save(os.path.join(seq_save_dir, str(frame_num)), landmarks)

                # Progress bar
                progress = int((frame_num + 1) / SEQUENCE_LENGTH * 200)
                cv2.rectangle(frame,
                              (10, frame.shape[0] - 30),
                              (10 + progress, frame.shape[0] - 10),
                              (0, 255, 100), -1)
                cv2.putText(frame,
                            f"Recording {frame_num+1}/{SEQUENCE_LENGTH}",
                            (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 255), 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            print(f"  ✓ Sequence {seq_idx} saved ({SEQUENCE_LENGTH} frames)")
            seq_idx   += 1
            collected += 1
            time.sleep(0.3)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f"\n[DONE] Collected {collected} sequences for '{word}'.")
    print(f"       Saved to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True,
                        help="Sign language word/gesture label")
    parser.add_argument("--sequences", type=int, default=30,
                        help="Number of sequences to collect (default: 30)")
    args = parser.parse_args()
    collect_word(args.word, args.sequences)