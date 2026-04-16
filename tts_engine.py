"""
Text-to-Speech Module
Wraps pyttsx3 (offline, zero-latency) with optional gTTS fallback.
Runs in a dedicated daemon thread to avoid blocking the main loop.
"""

import threading
import queue
import time

# Try pyttsx3 first (preferred: offline, low-latency)
try:
    import pyttsx3
    _PYTTSX3_OK = True
except ImportError:
    _PYTTSX3_OK = False

# gTTS fallback (requires internet)
try:
    from gtts import gTTS
    import tempfile, os
    try:
        import pygame
        _PYGAME_OK = True
    except ImportError:
        _PYGAME_OK = False
    _GTTS_OK = True
except ImportError:
    _GTTS_OK = False


class TTSEngine:
    """Thread-safe, non-blocking text-to-speech engine.

    Usage:
        tts = TTSEngine()
        tts.speak("Hello, how are you?")
        tts.close()     # graceful shutdown
    """

    def __init__(self, rate: int = 175, volume: float = 0.9,
                 voice_index: int = 0):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._backend = None

        if _PYTTSX3_OK:
            self._backend = "pyttsx3"
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
            voices = self._engine.getProperty("voices")
            if voices and voice_index < len(voices):
                self._engine.setProperty("voice", voices[voice_index].id)
            print(f"[TTS] Using pyttsx3 (rate={rate}, volume={volume})")
        elif _GTTS_OK:
            self._backend = "gtts"
            if _PYGAME_OK:
                pygame.mixer.init()
            print("[TTS] Using gTTS (requires internet)")
        else:
            print("[TTS] WARNING: No TTS library found. "
                  "Install pyttsx3 or gTTS.")

        # Dedicated worker thread
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="TTSWorker"
        )
        self._thread.start()

    # ── Public ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Queue a sentence for speech output. Non-blocking."""
        if text and text.strip():
            self._queue.put(text.strip())

    def close(self):
        """Gracefully stop the TTS worker."""
        self._running = False
        self._queue.put("")    # unblock worker
        self._thread.join(timeout=3)
        if self._backend == "pyttsx3":
            try:
                self._engine.stop()
            except Exception:
                pass

    # ── Worker ────────────────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            self._say(text)

    def _say(self, text: str):
        if self._backend == "pyttsx3":
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")

        elif self._backend == "gtts":
            try:
                tts_obj = gTTS(text=text, lang="en", slow=False)
                with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False) as fp:
                    tmp_path = fp.name
                tts_obj.save(tmp_path)
                if _PYGAME_OK:
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                else:
                    # Last-resort: system command
                    import subprocess
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", tmp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                os.unlink(tmp_path)
            except Exception as e:
                print(f"[TTS] gTTS error: {e}")
        else:
            print(f"[TTS] (no engine) Would say: {text}")
