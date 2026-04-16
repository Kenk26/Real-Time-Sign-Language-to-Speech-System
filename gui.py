"""
Step 5: Professional PyQt5 GUI Application
═══════════════════════════════════════════════════════════════════
Replaces the basic OpenCV UI with a sleek, dark-mode desktop application.
Runs the AI pipeline in a background thread to keep the UI perfectly smooth.

Usage:
    python 5_gui_app.py
"""

import sys
import cv2
import numpy as np
import tensorflow as tf
import json
import time
import threading
import os
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from emotion_detector import EmotionDetector
from llm_interpreter import LLMInterpreter
from tts_engine import TTSEngine

# ── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH   = 30
CONFIDENCE_THRESH = 0.70
PAUSE_SECONDS     = 2.5
MAX_TOKEN_BUFFER  = 20
MEDIAPIPE_MODEL_PATH = "hand_landmarker.task"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(5,17)
]

# ── Logic Helpers ────────────────────────────────────────────────────────────
def extract_landmarks(result) -> np.ndarray:
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)
    if result.hand_landmarks:
        for idx, hand_info in enumerate(result.handedness):
            label = hand_info[0].category_name 
            lms = result.hand_landmarks[idx]
            base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
            coords = []
            for lm in lms:
                coords.append(lm.x - base_x)
                coords.append(lm.y - base_y)
                coords.append(lm.z - base_z)
            coords_array = np.array(coords, dtype=np.float32)
            
            if label == "Left":
                if np.all(lh == 0): lh = coords_array
                else:               rh = coords_array
            else:
                if np.all(rh == 0): rh = coords_array
                else:               lh = coords_array
    return np.concatenate([lh, rh])

def normalize(seq: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(seq, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return seq / norms

def draw_skeleton(frame, lms, color=(0, 255, 180)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)


# ── AI Background Thread ─────────────────────────────────────────────────────
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_ui_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.flush_requested = False
        self.clear_requested = False
        
        # Load Model
        self.model = tf.keras.models.load_model("models/sign_model.keras")
        with open("models/model_meta.json") as f:
            meta = json.load(f)
        self.label_map = {int(k): v for k, v in meta["label_map"].items()}
        
        # Subsystems
        self.emotion_det = EmotionDetector(smoothing=12)
        self.llm = LLMInterpreter(model="gemma4:31b-cloud") # Set to your preferred model
        self.tts = TTSEngine()
        
        # State
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.token_buffer = []
        self.history = []
        self.current_sentence = ""
        self.sentence_lock = threading.Lock()
        self.llm_pending = False
        self.last_hand_time = time.time()
        
        # MediaPipe
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def on_llm_result(self, sentence: str):
        with self.sentence_lock:
            self.current_sentence = sentence
        self.token_buffer = []
        self.llm_pending = False
        self.tts.speak(sentence)
        if sentence:
            self.history.insert(0, sentence)
            if len(self.history) > 5:
                self.history.pop()

    def flush_to_llm(self):
        if not self.token_buffer or self.llm_pending: return
        self.llm_pending = True
        with self.sentence_lock:
            self.current_sentence = "Interpreting..."
        emotion = self.emotion_det.emotion
        self.llm.interpret_async(list(self.token_buffer), emotion, callback=self.on_llm_result)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cap.set(cv2.CAP_PROP_FPS, 60)
        
        last_word = None
        same_word_count = 0
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            # Handle manual button clicks from GUI
            if self.flush_requested:
                self.flush_to_llm()
                self.flush_requested = False
            if self.clear_requested:
                self.token_buffer.clear()
                self.frame_buffer.clear()
                with self.sentence_lock:
                    self.current_sentence = ""
                self.clear_requested = False

            # Emotion Detection
            emotion = self.emotion_det.update(frame)
            
            # Hand Tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.detector.detect(mp_img)
            
            hand_detected = bool(results.hand_landmarks)
            landmarks = extract_landmarks(results)
            
            if hand_detected:
                self.last_hand_time = time.time()
                for hand_lms in results.hand_landmarks:
                    draw_skeleton(frame, hand_lms, color=(0, 255, 180))
            
            self.frame_buffer.append(landmarks)
            confidence = 0.0
            
            # LSTM Prediction
            if len(self.frame_buffer) == SEQUENCE_LENGTH and hand_detected:
                seq = np.array(list(self.frame_buffer), dtype=np.float32)
                seq = normalize(seq)
                inp = np.expand_dims(seq, 0)
                probs = self.model.predict(inp, verbose=0)[0]
                idx = np.argmax(probs)
                confidence = float(probs[idx])
                
                if confidence >= CONFIDENCE_THRESH:
                    word = self.label_map[idx]
                    if word == last_word: same_word_count += 1
                    else: last_word = word; same_word_count = 1
                    
                    if same_word_count == 8:
                        self.token_buffer.append(word)
                        if len(self.token_buffer) >= MAX_TOKEN_BUFFER:
                            self.flush_to_llm()
                            
            # Auto Flush Pause Detection
            hand_gap = time.time() - self.last_hand_time
            if self.token_buffer and not self.llm_pending and hand_gap > PAUSE_SECONDS:
                self.flush_to_llm()

            # Emit clean video frame to GUI
            self.change_pixmap_signal.emit(frame)
            
            # Emit data to GUI
            with self.sentence_lock:
                current_sent = self.current_sentence
                
            self.update_ui_signal.emit({
                "emotion": emotion.upper(),
                "confidence": confidence,
                "tokens": " ".join(self.token_buffer),
                "sentence": current_sent,
                "history": self.history,
                "llm_busy": self.llm_pending
            })
            
        cap.release()
        self.detector.close()
        self.emotion_det.close()
        self.tts.close()

    def stop(self):
        self._run_flag = False
        self.wait()


# ── Main GUI Window ──────────────────────────────────────────────────────────
class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language to Speech AI")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #0f0f14; color: #f0f0f0;")
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # ── Left Side: Video & Subtitles ──
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel("Loading Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        self.video_label.setMinimumSize(960, 540)
        
        self.subtitle_label = QLabel("Waiting for gesture...")
        self.subtitle_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #00dca0; padding: 20px; background-color: #1a1a24; border-radius: 10px;")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setMinimumHeight(100)
        
        left_layout.addWidget(self.video_label, stretch=1)
        left_layout.addWidget(self.subtitle_label)
        
        # ── Right Side: Stats & Controls ──
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        
        # Stats Panel
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #1a1a24; border-radius: 10px; padding: 15px;")
        stats_layout = QVBoxLayout(stats_frame)
        
        self.emotion_label = QLabel("Emotion: NEUTRAL")
        self.emotion_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.conf_label = QLabel("Confidence: 0%")
        self.conf_label.setFont(QFont("Arial", 12))
        self.conf_bar = QProgressBar()
        self.conf_bar.setMaximum(100)
        self.conf_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #333; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #00dca0; width: 10px; }
        """)
        
        stats_layout.addWidget(self.emotion_label)
        stats_layout.addWidget(self.conf_label)
        stats_layout.addWidget(self.conf_bar)
        
        # Tokens Panel
        tokens_frame = QFrame()
        tokens_frame.setStyleSheet("background-color: #1a1a24; border-radius: 10px; padding: 15px;")
        tokens_layout = QVBoxLayout(tokens_frame)
        tokens_title = QLabel("Current Tokens:")
        tokens_title.setStyleSheet("color: #888;")
        self.tokens_label = QLabel("")
        self.tokens_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.tokens_label.setStyleSheet("color: #00a0ff;")
        self.tokens_label.setWordWrap(True)
        
        tokens_layout.addWidget(tokens_title)
        tokens_layout.addWidget(self.tokens_label)
        
        # History Panel
        history_frame = QFrame()
        history_frame.setStyleSheet("background-color: #1a1a24; border-radius: 10px; padding: 15px;")
        history_layout = QVBoxLayout(history_frame)
        history_title = QLabel("Conversation History:")
        history_title.setStyleSheet("color: #888;")
        self.history_label = QLabel("")
        self.history_label.setFont(QFont("Arial", 12))
        self.history_label.setWordWrap(True)
        self.history_label.setAlignment(Qt.AlignTop)
        
        history_layout.addWidget(history_title)
        history_layout.addWidget(self.history_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_flush = QPushButton("Translate Now (Space)")
        self.btn_flush.setStyleSheet("background-color: #00dca0; color: #000; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.btn_flush.clicked.connect(self.trigger_flush)
        
        self.btn_clear = QPushButton("Clear Buffer (C)")
        self.btn_clear.setStyleSheet("background-color: #cc4444; color: #fff; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.btn_clear.clicked.connect(self.trigger_clear)
        
        btn_layout.addWidget(self.btn_flush)
        btn_layout.addWidget(self.btn_clear)
        
        # Assemble Right Side
        right_layout.addWidget(stats_frame)
        right_layout.addWidget(tokens_frame)
        right_layout.addWidget(history_frame, stretch=1)
        right_layout.addLayout(btn_layout)
        
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)
        
        # ── Start AI Thread ──
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_ui_signal.connect(self.update_data)
        self.thread.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.trigger_flush()
        elif event.key() == Qt.Key_C:
            self.trigger_clear()
        elif event.key() == Qt.Key_Q:
            self.close()

    def trigger_flush(self):
        self.thread.flush_requested = True

    def trigger_clear(self):
        self.thread.clear_requested = True

    def update_image(self, cv_img):
        # Convert OpenCV BGR to Qt format
        qt_img = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        # Scale nicely to fit the label without distorting
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def update_data(self, data):
        # Emotion
        self.emotion_label.setText(f"Emotion: {data['emotion']}")
        if data['emotion'] == "HAPPY": self.emotion_label.setStyleSheet("color: #00dc64;")
        elif data['emotion'] == "SAD": self.emotion_label.setStyleSheet("color: #c86432;")
        elif data['emotion'] == "ANGRY": self.emotion_label.setStyleSheet("color: #003cdc;")
        else: self.emotion_label.setStyleSheet("color: #f0f0f0;")
        
        # Confidence
        conf_pct = int(data['confidence'] * 100)
        self.conf_label.setText(f"Confidence: {conf_pct}%")
        self.conf_bar.setValue(conf_pct)
        
        # Tokens
        self.tokens_label.setText(data['tokens'] if data['tokens'] else "(no tokens yet)")
        
        # Subtitle
        if data['llm_busy']:
            self.subtitle_label.setText("⟳ Interpreting...")
            self.subtitle_label.setStyleSheet("color: #00a0ff; padding: 20px; background-color: #1a1a24; border-radius: 10px;")
        elif data['sentence']:
            self.subtitle_label.setText(data['sentence'])
            self.subtitle_label.setStyleSheet("color: #00dca0; padding: 20px; background-color: #1a1a24; border-radius: 10px;")
        else:
            self.subtitle_label.setText("Waiting for gesture...")
            self.subtitle_label.setStyleSheet("color: #888; padding: 20px; background-color: #1a1a24; border-radius: 10px;")
            
        # History
        hist_text = "\n\n".join([f"• {h}" for h in data['history']])
        self.history_label.setText(hist_text)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())