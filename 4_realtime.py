"""
Step 4 + 5 + 6 + 7 + 8: Real-Time Sign Language → Speech System
═══════════════════════════════════════════════════════════════════
Pipeline:
  Camera → MediaPipe Hands/Face → Landmarks → LSTM → Tokens
         → Pause Detection → LLM → Subtitle → TTS

Usage:
    python 4_realtime.py
    python 4_realtime.py --model models/sign_model.keras
    python 4_realtime.py --threshold 0.75 --pause 2.0
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import threading
import argparse
import os
from collections import deque

from emotion_detector import EmotionDetector
from llm_interpreter import LLMInterpreter
from tts_engine import TTSEngine

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 30
CONFIDENCE_THRESH = 0.70
PAUSE_SECONDS    = 2.5      # silence window before LLM call
MAX_TOKEN_BUFFER = 20       # word tokens before forced flush
HISTORY_LINES    = 5        # conversation history shown on screen


# ── Landmark extraction (same as collection) ─────────────────────────────────
def extract_landmarks(results) -> np.ndarray:
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])
        lm -= lm[0]   # wrist-relative
        return lm.flatten()
    return np.zeros(63)


def normalize(seq: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(seq, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return seq / norms


# ── Drawing helpers ──────────────────────────────────────────────────────────
PALETTE = {
    "bg_panel":   (15,  15,  20),
    "accent":     (0,  220, 160),
    "accent2":    (0,  160, 255),
    "text_main":  (240, 240, 240),
    "text_dim":   (120, 120, 130),
    "warning":    (0,   90, 220),
    "good":       (0,  200, 100),
    "emotion":    (220, 180,   0),
}

EMOTION_COLORS = {
    "happy":     (0, 220, 100),
    "sad":       (200, 100,  50),
    "angry":     (0,  60,  220),
    "surprised": (220, 180,  0),
    "neutral":   (140, 140, 150),
}

EMOTION_ICONS = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "surprised": "😲", "neutral": "😐",
}


def draw_panel(frame, x, y, w, h, alpha=0.55, color=(15, 15, 20)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_text(frame, text, pos, font_scale=0.6, color=(240, 240, 240),
              thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)


def draw_confidence_bar(frame, x, y, confidence, w=160, h=8):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 50), -1)
    bar_w = int(w * confidence)
    color = PALETTE["good"] if confidence > 0.8 else \
            PALETTE["accent"] if confidence > CONFIDENCE_THRESH else \
            (100, 100, 120)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + h), color, -1)


# ── Main system ──────────────────────────────────────────────────────────────
def run(model_path: str = "models/sign_model.keras",
        meta_path: str = "models/model_meta.json",
        conf_threshold: float = CONFIDENCE_THRESH,
        pause_sec: float = PAUSE_SECONDS,
        llm_model: str = "gemma3:1b"):

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at '{model_path}'.")
        print("        Run: python 3_train_model.py   first.")
        return

    print("[INFO] Loading LSTM model …")
    model = tf.keras.models.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)
    label_map: dict = {int(k): v for k, v in meta["label_map"].items()}
    num_classes = meta["num_classes"]
    print(f"[INFO] Classes: {list(label_map.values())}")

    # ── Subsystems ────────────────────────────────────────────────────────
    emotion_det = EmotionDetector(smoothing=12)
    llm = LLMInterpreter(model=llm_model)
    tts = TTSEngine()

    # ── State ─────────────────────────────────────────────────────────────
    frame_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    token_buffer: list[str] = []
    conversation_history: list[str] = []
    last_prediction_time = time.time()
    last_word = None
    same_word_count = 0
    SAME_WORD_THRESH = 8

    current_sentence = ""
    sentence_lock = threading.Lock()
    llm_pending = False
    last_hand_time = time.time()
    fps_times: deque = deque(maxlen=30)

    def on_llm_result(sentence: str):
        nonlocal current_sentence, llm_pending, token_buffer
        with sentence_lock:
            current_sentence = sentence
        token_buffer = []
        llm_pending = False
        tts.speak(sentence)
        if sentence:
            conversation_history.append(sentence)
            if len(conversation_history) > HISTORY_LINES:
                conversation_history.pop(0)
        print(f"[LLM] → {sentence}")

    def flush_to_llm():
        nonlocal llm_pending, current_sentence
        if not token_buffer or llm_pending:
            return
        llm_pending = True
        with sentence_lock:
            current_sentence = "…"
        emotion = emotion_det.emotion
        llm.interpret_async(
            list(token_buffer), emotion, callback=on_llm_result
        )

    # ── MediaPipe ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        print("\n[INFO] Real-time inference started.")
        print("       SPACE → flush tokens to LLM | Q → quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ── Emotion ───────────────────────────────────────────────
            emotion = emotion_det.update(frame)

            # ── Hand landmark extraction ──────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_detected = bool(results.multi_hand_landmarks)
            landmarks = extract_landmarks(results)

            if hand_detected:
                last_hand_time = time.time()
                # Draw landmarks
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=PALETTE["accent"], thickness=2,
                            circle_radius=4),
                        mp_drawing.DrawingSpec(
                            color=(200, 200, 200), thickness=1),
                    )

            frame_buffer.append(landmarks)

            # ── LSTM prediction ───────────────────────────────────────
            predicted_word = None
            confidence = 0.0

            if (len(frame_buffer) == SEQUENCE_LENGTH
                    and hand_detected):
                seq = np.array(list(frame_buffer),
                                dtype=np.float32)  # (30, 63)
                seq = normalize(seq)
                inp = np.expand_dims(seq, 0)          # (1, 30, 63)
                probs = model.predict(inp, verbose=0)[0]
                idx = np.argmax(probs)
                confidence = float(probs[idx])

                if confidence >= conf_threshold:
                    word = label_map[idx]

                    # De-duplicate: only add if held for N frames
                    if word == last_word:
                        same_word_count += 1
                    else:
                        last_word = word
                        same_word_count = 1

                    if same_word_count == SAME_WORD_THRESH:
                        token_buffer.append(word)
                        last_prediction_time = time.time()
                        predicted_word = word
                        print(f"  [{confidence:.2f}] {word}")

                        if len(token_buffer) >= MAX_TOKEN_BUFFER:
                            flush_to_llm()

            # ── Pause detection → auto flush ──────────────────────────
            hand_gap = time.time() - last_hand_time
            if (token_buffer and not llm_pending
                    and hand_gap > pause_sec):
                flush_to_llm()

            # ─────────────────────────────────────────────────────────
            # UI RENDERING
            # ─────────────────────────────────────────────────────────

            fps_times.append(time.time())
            fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-5) \
                  if len(fps_times) > 1 else 0

            # Top bar
            draw_panel(frame, 0, 0, w, 54)
            draw_text(frame, "Sign Language → Speech  |  AI System",
                      (16, 34), 0.75, PALETTE["accent"], 2)
            draw_text(frame, f"FPS: {fps:.1f}",
                      (w - 110, 34), 0.6, PALETTE["text_dim"])

            # Emotion badge
            em_color = EMOTION_COLORS.get(emotion, PALETTE["text_dim"])
            draw_panel(frame, w - 200, 60, 190, 44, 0.65)
            draw_text(frame, f"Emotion: {emotion.upper()}",
                      (w - 194, 88), 0.6, em_color, 2)

            # Confidence bar (bottom of video area, right)
            if hand_detected:
                draw_panel(frame, w - 220, h - 80, 210, 60, 0.6)
                draw_text(frame, f"Conf: {confidence:.2f}",
                          (w - 214, h - 55), 0.55, PALETTE["text_main"])
                draw_confidence_bar(frame, w - 214, h - 38, confidence, 190)

            # Token buffer display
            panel_h = 56
            draw_panel(frame, 0, h - panel_h - 80, w // 2 + 40, panel_h, 0.7)
            tokens_display = " ".join(token_buffer[-10:]) or "(no tokens yet)"
            draw_text(frame, "Tokens:", (10, h - panel_h - 54),
                      0.5, PALETTE["text_dim"])
            draw_text(frame, tokens_display, (10, h - panel_h - 28),
                      0.65, PALETTE["accent2"], 2)

            # Sentence subtitle
            sub_panel_y = h - 76
            draw_panel(frame, 0, sub_panel_y, w, 76, 0.75)
            cv2.line(frame, (0, sub_panel_y + 2), (w, sub_panel_y + 2),
                     PALETTE["accent"], 1)
            with sentence_lock:
                sentence_display = current_sentence

            if sentence_display:
                # Word-wrap long sentences
                words = sentence_display.split()
                lines, line = [], []
                for word_i in words:
                    line.append(word_i)
                    if len(" ".join(line)) > 55:
                        lines.append(" ".join(line[:-1]))
                        line = [word_i]
                if line:
                    lines.append(" ".join(line))

                for li, ltext in enumerate(lines[:2]):
                    draw_text(frame, ltext,
                              (16, sub_panel_y + 30 + li * 30),
                              0.8, PALETTE["text_main"], 2)
            else:
                draw_text(frame, "(waiting for gesture…)",
                          (16, sub_panel_y + 38),
                          0.7, PALETTE["text_dim"])

            # LLM busy indicator
            if llm_pending:
                draw_text(frame, "⟳ Interpreting…",
                          (w - 230, sub_panel_y + 38),
                          0.6, PALETTE["accent"])

            # Conversation history (right panel)
            if conversation_history:
                hist_x = w - 340
                draw_panel(frame, hist_x, 110,
                           330, len(conversation_history) * 28 + 20, 0.6)
                draw_text(frame, "History",
                          (hist_x + 8, 132), 0.5, PALETTE["text_dim"])
                for i, hist_line in enumerate(conversation_history):
                    short = hist_line[:38] + "…" \
                            if len(hist_line) > 38 else hist_line
                    draw_text(frame, f"• {short}",
                              (hist_x + 8, 155 + i * 26),
                              0.48, PALETTE["text_main"])

            # Hand detection status dot
            dot_color = PALETTE["good"] if hand_detected else (80, 80, 90)
            cv2.circle(frame, (w - 24, h - panel_h - 100), 7, dot_color, -1)

            cv2.imshow("Sign Language → Speech", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                flush_to_llm()
            elif key == ord("c"):
                # Clear buffers
                token_buffer.clear()
                frame_buffer.clear()
                with sentence_lock:
                    current_sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    emotion_det.close()
    tts.close()
    print("\n[INFO] System shut down cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="models/sign_model.keras")
    parser.add_argument("--meta",
                        default="models/model_meta.json")
    parser.add_argument("--threshold", type=float,
                        default=CONFIDENCE_THRESH)
    parser.add_argument("--pause", type=float,
                        default=PAUSE_SECONDS)
    parser.add_argument("--llm-model", default="gemma3:1b",
                        help="Ollama model name")
    args = parser.parse_args()

    run(
        model_path=args.model,
        meta_path=args.meta,
        conf_threshold=args.threshold,
        pause_sec=args.pause,
        llm_model=args.llm_model,
    )
