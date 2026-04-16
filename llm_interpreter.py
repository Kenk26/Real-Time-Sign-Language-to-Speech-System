"""
LLM Integration Module
Uses LangChain + Ollama (gemma:27b or gemma4:31b-cloud) to convert
raw gesture tokens + emotion into a natural, grammatically-correct sentence.
"""

import threading
from typing import Callable, Optional

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain_ollama import ChatOllama                      # ✅ correct class
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   # ✅ for runnables
# LLMChain removed — replaced by LCEL pipe (prompt | llm | parser)

# ── Prompt template ──────────────────────────────────────────────────────────
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["tokens", "emotion"],
    template="""You are a sign language interpreter assistant.
Convert the following sign language gesture tokens into a single, natural English sentence.

Words: {tokens}
Emotion: {emotion}

Rules:
- Fix grammar and word order
- Maintain the original meaning
- Reflect the emotion subtly in tone if appropriate
- Return ONLY the final sentence, nothing else
- Do not add explanations, quotes, or punctuation beyond the sentence

Sentence:"""
)


class LLMInterpreter:
    """Wraps Ollama LLM for async sentence interpretation.

    Usage:
        llm = LLMInterpreter()
        # Async (non-blocking):
        llm.interpret_async(["hello", "how", "you"],
                            emotion="happy",
                            callback=lambda s: print(s))
        # Sync:
        sentence = llm.interpret(["hello", "how", "you"], "happy")
    """

    def __init__(self,
                 model: str = "gemma3:1b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.4,
                 timeout: int = 30):
        """
        Args:
            model:      Ollama model name. Examples:
                        "gemma3:1b"          ← lightweight for dev/testing
                        "gemma:7b"           ← good balance
                        "gemma4:31b-cloud"   ← as specified in project brief
            base_url:   Ollama server URL (default local).
            temperature: Generation temperature.
            timeout:    Request timeout in seconds.
        """
        self._model_name = model
        self._lock = threading.Lock()
        self._busy = False

        try:
            llm = ChatOllama(                                # ✅ ChatOllama
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
            )
            # ✅ LCEL runnable chain — replaces deprecated LLMChain
            self._chain = PROMPT_TEMPLATE | llm | StrOutputParser()
            print(f"[LLMInterpreter] Connected to Ollama model '{model}'.")
        except Exception as e:
            print(f"[LLMInterpreter] WARNING: Could not connect to Ollama "
                  f"({e}). Using fallback rule-based interpreter.")
            self._chain = None

    # ── Public API ────────────────────────────────────────────────────────

    def interpret(self, tokens: list[str], emotion: str = "neutral") -> str:
        """Synchronously convert tokens → sentence. Blocks until done."""
        if not tokens:
            return ""

        token_str = " ".join(tokens)

        if self._chain is None:
            return self._fallback(tokens, emotion)

        try:
            # ✅ Runnable returns str directly via StrOutputParser
            sentence = self._chain.invoke(
                {"tokens": token_str, "emotion": emotion}
            ).strip().strip('"').strip("'")
            return sentence if sentence else self._fallback(tokens, emotion)
        except Exception as e:
            print(f"[LLMInterpreter] Inference error: {e}")
            return self._fallback(tokens, emotion)

    def interpret_async(self,
                        tokens: list[str],
                        emotion: str = "neutral",
                        callback: Optional[Callable[[str], None]] = None):
        """Non-blocking interpretation. Result delivered via callback."""
        if self._busy:
            return

        def _run():
            with self._lock:
                self._busy = True
            try:
                sentence = self.interpret(tokens, emotion)
            finally:
                self._busy = False
            if callback:
                callback(sentence)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    @property
    def is_busy(self) -> bool:
        return self._busy

    # ── Fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(tokens: list[str], emotion: str) -> str:
        """Simple heuristic when Ollama is unavailable."""
        sentence = " ".join(tokens).strip()
        if not sentence:
            return ""
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith((".", "?", "!")):
            punctuation = "!" if emotion == "happy" else "."
            sentence += punctuation
        return sentence