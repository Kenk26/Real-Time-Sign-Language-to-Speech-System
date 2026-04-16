# 🤟 Sign Language to Speech System

A real-time sign language recognition pipeline that captures hand gestures via webcam, classifies them using a deep learning model, interprets gesture sequences into natural English sentences via an LLM, and speaks them aloud using text-to-speech.

---

## ✨ Features

- **Real-time hand tracking** using MediaPipe Tasks API (Hand Landmarker)
- **Facial emotion detection** using MediaPipe Face Landmarker with optional DeepFace backend
- **LSTM-based gesture classifier** trained on custom-collected sequences
- **LLM-powered sentence construction** via LangChain + Ollama (Gemma)
- **Text-to-speech output** using `pyttsx3` (offline) with `gTTS` fallback
- **Interactive GUI** for live inference and feedback
- **Full data collection + training pipeline** included

---

## 🗂️ Project Structure

```
sign-language-to-speech/
│
├── collect_data.py          # Step 1 — Webcam data collection
├── preprocess.py            # Step 2 — Landmark normalization & label encoding
├── training_notebook.ipynb  # Step 3 — LSTM model training & evaluation
│
├── emotion_detector.py      # Facial emotion detection module
├── llm_interpreter.py       # LLM-based gesture-to-sentence module
├── tts_engine.py            # Text-to-speech engine (pyttsx3 / gTTS)
├── gui.py                   # Main application GUI
│
├── dataset/                 # Auto-created during data collection
├── processed/               # Auto-created during preprocessing
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- **Python 3.10.8**
- Webcam
- [Ollama](https://ollama.com/) running locally (for LLM sentence interpretation)

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** To enable the richer DeepFace emotion backend, uncomment `deepface` in `requirements.txt` before installing (it is included by default in the current file).

### Pull the Ollama model

```bash
ollama pull gemma4:31b-cloud # full model as per project specification
```

---

## 🚀 Quickstart

### 1 — Collect gesture data

Record sequences for each word/sign you want the model to recognise:

```bash
python collect_data.py --word "hello"  --sequences 30
python collect_data.py --word "thanks" --sequences 30
python collect_data.py --word "yes"    --sequences 30
```

- Press **Space** to begin recording each sequence.
- Press **Q** to quit early.
- Each sequence captures **30 frames** of hand landmarks and saves them to `dataset/<word>/`.

### 2 — Preprocess the dataset

```bash
python preprocess.py
```

Outputs to `processed/`:
| File | Description |
|---|---|
| `X_train.npy` | Training sequences `(N, 30, 126)` |
| `X_test.npy` | Test sequences |
| `y_train.npy` | One-hot encoded labels |
| `y_test.npy` | One-hot encoded labels |
| `label_map.json` | Index → word mapping |

### 3 — Train the model

Open and run `training_notebook.ipynb` in Jupyter:

```bash
jupyter notebook training_notebook.ipynb
```

### 4 — Launch the application

```bash
python gui.py
```

---

## 🧠 System Architecture

```
Webcam Feed
    │
    ├──▶ HandLandmarker (MediaPipe Tasks API)
    │         └──▶ 126-dim landmark vector (Left + Right hand)
    │                   └──▶ LSTM Classifier ──▶ Gesture Token
    │
    ├──▶ FaceLandmarker (MediaPipe Tasks API)
    │         └──▶ Geometric heuristics / DeepFace
    │                   └──▶ Emotion Label
    │
    └──▶ LLMInterpreter (LangChain + Ollama / Gemma)
              └──▶ Natural English Sentence
                        └──▶ TTSEngine (pyttsx3 / gTTS)
                                  └──▶ 🔊 Speech Output
```

---

## 📐 Landmark Format

Each frame is represented as a **126-dimensional vector**:
- **[0–62]** — Left hand: 21 landmarks × 3 (x, y, z), normalized relative to wrist
- **[63–125]** — Right hand: 21 landmarks × 3 (x, y, z), normalized relative to wrist

Missing hands are zero-padded.

---

## 🎭 Emotion Detection

The `EmotionDetector` class supports two backends:

| Backend | Description | Requirement |
|---|---|---|
| **MediaPipe heuristics** (default) | Geometric ratios from face landmarks | None |
| **DeepFace** | Deep learning emotion classifier | `pip install deepface` |

Detected emotions: `neutral`, `happy`, `sad`, `angry`, `surprised`

---

## 🗣️ LLM Interpretation

`LLMInterpreter` converts a list of gesture tokens and an emotion into a grammatically correct sentence:

```python
from llm_interpreter import LLMInterpreter

llm = LLMInterpreter(model="gemma4:31b-cloud")

# Synchronous
sentence = llm.interpret(["hello", "how", "you"], emotion="happy")
# → "Hello! How are you?"

# Asynchronous (non-blocking)
llm.interpret_async(["thank", "you"], emotion="neutral",
                    callback=lambda s: print(s))
```

If Ollama is unavailable, the module falls back to a simple rule-based capitalizer automatically.

---

## 🔊 TTS Engine

`TTSEngine` queues speech output in a background daemon thread — it never blocks the main loop.

```python
from tts_engine import TTSEngine

tts = TTSEngine(rate=175, volume=0.9)
tts.speak("Hello, how are you?")
tts.close()
```

| Backend | Requirement | Network |
|---|---|---|
| `pyttsx3` (preferred) | `pip install pyttsx3` | Offline |
| `gTTS` (fallback) | `pip install gTTS pygame` | Online |

---

## 🛠️ Configuration

Key constants can be adjusted at the top of each module:

| Constant | File | Default | Description |
|---|---|---|---|
| `SEQUENCE_LENGTH` | `collect_data.py`, `preprocess.py` | `30` | Frames per gesture |
| `DATASET_DIR` | `collect_data.py`, `preprocess.py` | `"dataset"` | Raw data path |
| `PROCESSED_DIR` | `preprocess.py` | `"processed"` | Processed data path |
| `model` | `llm_interpreter.py` | `"gemma4:31b-cloud"` | Ollama model name |
| `smoothing` | `emotion_detector.py` | `10` | Emotion smoothing window |

---

## 🐛 Troubleshooting

**Webcam not opening**
```
RuntimeError: Cannot open webcam.
```
Check that no other application is using the camera and that your device index is correct (default is `0`).

**MediaPipe model not found**
The `.task` model files are downloaded automatically on first run. Ensure you have an active internet connection during the initial launch.

**Ollama not responding**
Make sure the Ollama daemon is running:
```bash
ollama serve
```
The system will fall back to rule-based interpretation if Ollama is unreachable.

**pyttsx3 errors on Windows**
The engine is initialized inside its worker thread to avoid COM threading issues. If problems persist, try installing `pywin32`:
```bash
pip install pywin32
```

---

## 📄 License

This project is licensed under the **MIT License**.
