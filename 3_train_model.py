"""
Step 3: LSTM Model Training

Architecture:
  Input  → (30, 63)
  LSTM   → 64 units, return_sequences=True
  LSTM   → 64 units
  Dense  → 64, ReLU  + Dropout
  Output → Softmax (num_classes)

Usage:
    python 3_train_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

PROCESSED_DIR = "processed"
MODEL_DIR = "models"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def build_model(sequence_length: int, feature_dim: int,
                num_classes: int) -> tf.keras.Model:
    model = Sequential([
        # Layer 1 – LSTM with return_sequences
        LSTM(64, return_sequences=True,
             input_shape=(sequence_length, feature_dim),
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),

        # Layer 2 – LSTM
        LSTM(64, return_sequences=False,
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),

        # Dense head
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),

        # Output
        Dense(num_classes, activation="softmax"),
    ], name="SignLanguageLSTM")
    return model


def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load processed data ───────────────────────────────────────────────
    print("[1/4] Loading processed data …")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    with open(os.path.join(PROCESSED_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    seq_len, feat_dim = X_train.shape[1], X_train.shape[2]

    print(f"      X_train: {X_train.shape}  |  Classes: {num_classes}")

    # ── Build model ───────────────────────────────────────────────────────
    print("[2/4] Building model …")
    model = build_model(seq_len, feat_dim, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    best_weights_path = os.path.join(MODEL_DIR, "best_model.weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_weights_path, monitor="val_accuracy",
                        save_best_only=True, save_weights_only=True,
                        verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=os.path.join(MODEL_DIR, "logs")),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print("[3/4] Training …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate & save ───────────────────────────────────────────────────
    print("[4/4] Evaluating …")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"      Test accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # Save full model (SavedModel format)
    saved_path = os.path.join(MODEL_DIR, "sign_model.keras")
    model.save(saved_path)
    print(f"      Model saved → {saved_path}")

    # Save training curves
    plot_history(history, os.path.join(MODEL_DIR, "training_curves.png"))

    # Save metadata
    meta = {
        "sequence_length": int(seq_len),
        "feature_dim": int(feat_dim),
        "num_classes": int(num_classes),
        "label_map": label_map,
        "test_accuracy": float(acc),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Model training complete.")
    print(f"       Saved to: {os.path.abspath(MODEL_DIR)}/")


if __name__ == "__main__":
    train()
