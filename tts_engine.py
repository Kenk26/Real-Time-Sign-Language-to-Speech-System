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
    """Thread-safe, non-blocking text-to-speech engine."""

    def __init__(self, rate: int = 175, volume: float = 0.9,
                 voice_index: int = 0):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = True
        
        # Save settings to apply later inside the thread
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index

        if _PYTTSX3_OK:
            self._backend = "pyttsx3"
            print(f"[TTS] pyttsx3 selected (rate={rate}, volume={volume})")
        elif _GTTS_OK:
            self._backend = "gtts"
            print("[TTS] Using gTTS (requires internet)")
        else:
            self._backend = None
            print("[TTS] WARNING: No TTS library found. Install pyttsx3 or gTTS.")

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

    # ── Worker ────────────────────────────────────────────────────────────

    def _worker(self):
        # ⚠️ Windows Fix: COM objects (pyttsx3) MUST be initialized 
        # inside the exact thread that uses them!
        engine = None
        
        if self._backend == "pyttsx3":
            # CoInitialize is required for COM multithreading on Windows
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
                
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            voices = engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                engine.setProperty("voice", voices[self._voice_index].id)
                
        elif self._backend == "gtts":
            if _PYGAME_OK:
                pygame.mixer.init()

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
                
            if self._backend == "pyttsx3":
                try:
                    engine.say(text)
                    engine.runAndWait()
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