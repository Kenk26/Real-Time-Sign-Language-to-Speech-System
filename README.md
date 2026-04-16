# Real-Time Sign Language → Speech System
### Gesture Recognition · Emotion Detection · LLM Interpretation · TTS

---

## Architecture

```
Webcam
  │
  ▼
MediaPipe (Hands 21pts + Face Mesh)
  │
  ├─► Landmark Extraction (63 features)
  │     └─► Sliding Window Buffer (30 frames)
  │           └─► LSTM Model → Word Token + Confidence
  │
  └─► Emotion Detector → happy / sad / neutral / angry / surprised
  
Word Tokens ──┐
Emotion      ─┤──► LangChain + Ollama (gemma4:31b-cloud)
              │         └─► Natural Sentence
              │
              ├──► Subtitle Overlay (OpenCV UI)
              └──► Text-to-Speech (pyttsx3 / gTTS)
```

---

## Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Pull Ollama model
```bash
# Lightweight (good for dev):
ollama pull gemma3:1b

# Project spec (requires capable GPU / cloud):
ollama pull gemma4:31b-cloud
```

### 3 · Collect training data
```bash
# Collect 30 sequences of "hello"
python 1_collect_data.py --word hello --sequences 30

# Collect more words
python 1_collect_data.py --word thanks --sequences 30
python 1_collect_data.py --word yes    --sequences 30
python 1_collect_data.py --word no     --sequences 30
python 1_collect_data.py --word help   --sequences 30
```
*Press **SPACE** before each recording. Dataset saved to `dataset/<word>/<seq_id>/<frame>.npy`*

### 4 · Preprocess
```bash
python 2_preprocess.py
```
Outputs: `processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `label_map.json`

### 5 · Train LSTM model
```bash
python 3_train_model.py
```
Outputs: `models/sign_model.keras`, `models/model_meta.json`, `models/training_curves.png`

Or use the Jupyter notebook:
```bash
jupyter notebook training_notebook.ipynb
```

### 6 · Run real-time system
```bash
# With default lightweight LLM:
python 4_realtime.py

# With project-spec model:
python 4_realtime.py --llm-model gemma4:31b-cloud

# Custom confidence / pause:
python 4_realtime.py --threshold 0.75 --pause 2.0
```

---

## Keyboard Controls (real-time mode)

| Key | Action |
|-----|--------|
| `SPACE` | Flush current token buffer to LLM immediately |
| `C` | Clear tokens and current sentence |
| `Q` | Quit |

---

## File Structure

```
sign_language_system/
├── 1_collect_data.py       # webcam data collection
├── 2_preprocess.py         # normalize + encode + split
├── 3_train_model.py        # LSTM training script
├── 4_realtime.py           # main real-time inference
├── emotion_detector.py     # MediaPipe / DeepFace emotion module
├── llm_interpreter.py      # LangChain + Ollama integration
├── tts_engine.py           # pyttsx3 / gTTS TTS engine
├── training_notebook.ipynb # Jupyter training notebook
├── requirements.txt        # Python dependencies
│
├── dataset/                # created by 1_collect_data.py
│   ├── hello/
│   │   ├── 0/   0.npy … 29.npy
│   │   └── 1/   …
│   └── thanks/  …
│
├── processed/              # created by 2_preprocess.py
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── label_map.json
│
└── models/                 # created by 3_train_model.py
    ├── sign_model.keras
    ├── model_meta.json
    ├── best_model.weights.h5
    └── training_curves.png
```

---

## Model Details

| Property | Value |
|----------|-------|
| Input shape | `(30, 63)` — 30 frames × 21 landmarks × 3 coords |
| Layer 1 | LSTM(64, return_sequences=True) + BatchNorm |
| Layer 2 | LSTM(64) + BatchNorm |
| Dense | 64 ReLU → Dropout(0.3) → 32 ReLU → Dropout(0.2) |
| Output | Softmax(num_classes) |
| Optimizer | Adam(lr=1e-3) |
| Loss | categorical_crossentropy |
| Epochs | 50 (EarlyStopping patience=10) |
| Batch | 16 |

---

## LLM Prompt Template

```
Convert sign language gesture tokens into a single, natural English sentence.

Words: {tokens}
Emotion: {emotion}

Rules:
- Fix grammar and word order
- Maintain the original meaning
- Reflect the emotion subtly in tone if appropriate
- Return ONLY the final sentence
```

---

## Performance Tips

- **GPU**: TensorFlow will auto-use CUDA if available
- **Confidence threshold**: Lower `--threshold` (e.g. `0.65`) if predictions miss; raise if noisy
- **Pause detection**: Adjust `--pause` to match your signing speed
- **LLM latency**: Use `gemma3:1b` on CPU; `gemma4:31b-cloud` needs a GPU or cloud endpoint
- **Threading**: LLM calls run in daemon threads — UI never blocks

---

## Extending

### Add new gestures
```bash
python 1_collect_data.py --word <new_word> --sequences 40
python 2_preprocess.py
python 3_train_model.py
```

### Enable DeepFace emotions
```bash
pip install deepface
```
Then in `4_realtime.py`, pass `use_deepface=True` to `EmotionDetector`.

### Conversation history
`conversation_history` list in `4_realtime.py` keeps the last 5 LLM outputs and displays them in the overlay.
