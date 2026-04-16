"""
Step 2: Preprocessing
Loads .npy sequences, normalizes landmarks, encodes labels,
splits into train/test sets, and saves processed arrays.

Usage:
    python 2_preprocess.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

DATASET_DIR = "dataset"
SEQUENCE_LENGTH = 30
PROCESSED_DIR = "processed"


def load_dataset(dataset_dir: str, sequence_length: int = 30):
    """Walk dataset/ and load all sequences into X, y arrays."""
    X, y = [], []
    label_map = {}

    words = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not words:
        raise ValueError(f"No word folders found in '{dataset_dir}'. "
                         "Run 1_collect_data.py first.")

    print(f"[INFO] Found words: {words}")

    for word in words:
        word_dir = os.path.join(dataset_dir, word)
        sequences = sorted([
            d for d in os.listdir(word_dir)
            if os.path.isdir(os.path.join(word_dir, d))
        ])

        for seq_id in sequences:
            seq_dir = os.path.join(word_dir, seq_id)
            frames = []

            for frame_idx in range(sequence_length):
                frame_path = os.path.join(seq_dir, f"{frame_idx}.npy")
                if os.path.exists(frame_path):
                    frames.append(np.load(frame_path))
                else:
                    frames.append(np.zeros(63))   # pad missing frame

            if len(frames) == sequence_length:
                X.append(frames)
                y.append(word)

        print(f"  '{word}': {len(sequences)} sequences loaded")

    X = np.array(X, dtype=np.float32)  # (N, 30, 63)
    y = np.array(y)
    return X, y, words


def normalize_sequences(X: np.ndarray) -> np.ndarray:
    """Per-sequence min-max normalization across spatial dims."""
    # X shape: (N, T, 63)
    # Already wrist-relative from collection; add global scale normalization
    norms = np.linalg.norm(X, axis=-1, keepdims=True)  # (N, T, 1)
    norms = np.where(norms == 0, 1, norms)              # avoid div-by-zero
    return X / norms


def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    X, y_raw, words = load_dataset(DATASET_DIR, SEQUENCE_LENGTH)
    print(f"      Loaded {len(X)} sequences, shape: {X.shape}")

    # ── Normalize ────────────────────────────────────────────────────────────
    print("[2/4] Normalizing landmarks …")
    X = normalize_sequences(X)

    # ── Encode labels ────────────────────────────────────────────────────────
    print("[3/4] Encoding labels …")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)               # integers
    from tensorflow.keras.utils import to_categorical  # noqa: E402
    y_cat = to_categorical(y_encoded, num_classes=len(words))

    # Save label mapping
    label_map = {int(i): word for i, word in enumerate(le.classes_)}
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"      Classes: {list(le.classes_)}")

    # ── Train / test split ───────────────────────────────────────────────────
    print("[4/4] Splitting train/test (80/20) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"\n[DONE] Preprocessed data saved to '{PROCESSED_DIR}/'")
    print(f"       X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"       y_train: {y_train.shape}  |  y_test: {y_test.shape}")
    print(f"       Label map → {PROCESSED_DIR}/label_map.json")


if __name__ == "__main__":
    preprocess()
