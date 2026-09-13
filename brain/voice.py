"""
brain/voice.py — Real-time Speechmatics ASR with settling buffer.

Uses the speechmatics-rt SDK (modern replacement for speechmatics-python).
Settling buffer tau = 0.774s (p95 empirical settling time).

Usage:
    # Text-queue mode (no microphone needed for testing):
    voice = SpeechmaticsVoiceInput(api_key=os.environ['SPEECHMATICS_API_KEY'])
    voice.start(on_instruction=lambda text: print("Instruction:", text))
    voice.send_text("set the table")  # inject text instruction
    voice.stop()
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

# ─── Settling Buffer ──────────────────────────────────────────────────────────

class TranscriptBuffer:
    """Accumulates partial Speechmatics transcripts and fires complete utterances.

    A word is 'settled' when it hasn't changed for tau seconds. This prevents
    the pipeline from acting on mid-word partial results.
    """

    def __init__(self, tau: float = 0.774) -> None:
        self.tau = tau
        self._parts: list[dict] = []       # raw partial results
        self._settled_words: list[str] = []
        self._last_partial: str = ""
        self._last_update: float = time.monotonic()
        self._pending_flush: str = ""

    def on_partial(self, text: str) -> None:
        """Call with each partial transcript token."""
        self._last_partial = text
        self._last_update = time.monotonic()

    def on_final(self, text: str) -> Optional[str]:
        """Call with each final transcript token. Returns settled utterance or None."""
        now = time.monotonic()
        self._pending_flush = (self._pending_flush + " " + text).strip()
        if now - self._last_update >= self.tau:
            result = self._pending_flush
            self._pending_flush = ""
            self._last_update = now
            return result if result else None
        self._last_update = now
        return None

    def flush(self) -> Optional[str]:
        """Force-flush any pending text (call when utterance ends)."""
        if self._pending_flush:
            result = self._pending_flush
            self._pending_flush = ""
            return result
        return None

    def get_partial(self) -> str:
        return self._last_partial


# ─── NLP Router ───────────────────────────────────────────────────────────────

class NLPRouter:
    """Classifies settled utterances without an external model.

    Returns (classification, cleaned_instruction) where classification is:
      'new_instruction'     — completely new task
      'mid_task_correction' — modify current task
      'query'               — question about current state
      'unknown'             — unclassified
    """

    _NEW_TASK_KEYWORDS = {
        "set", "place", "put", "lay", "open", "pick", "grab", "move",
        "arrange", "bring", "fetch", "take", "start", "begin", "do",
    }
    _CORRECTION_KEYWORDS = {
        "no", "wait", "stop", "actually", "instead", "not that",
        "other side", "wrong", "undo", "cancel", "change",
    }
    _QUERY_KEYWORDS = {
        "what", "where", "which", "how", "status", "done", "finished",
        "progress", "check",
    }

    def classify(self, text: str, current_phase: str = "") -> tuple[str, str]:
        """Classify text into one of 4 categories. Returns (category, cleaned)."""
        cleaned = text.strip().lower()
        words = set(cleaned.split())

        if words & self._QUERY_KEYWORDS:
            return ("query", text.strip())

        if words & self._CORRECTION_KEYWORDS:
            return ("mid_task_correction", text.strip())

        if words & self._NEW_TASK_KEYWORDS:
            return ("new_instruction", text.strip())

        # Fallback: if we're not in a task and text is long enough, treat as instruction
        if len(cleaned.split()) >= 3 and current_phase in ("", "done", "home"):
            return ("new_instruction", text.strip())

        return ("unknown", text.strip())


# ─── Text-Queue Input (fallback when microphone unavailable) ──────────────────

class TextQueueInput:
    """Simulates real-time voice input via a queue. Use for testing."""

    def __init__(self, on_instruction: Callable[[str], None]) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._on_instruction = on_instruction
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def send(self, text: str) -> None:
        """Inject a text instruction (simulates voice input)."""
        self._queue.put(text)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
                self._on_instruction(text)
            except queue.Empty:
                continue


# ─── Main Voice Input Class ───────────────────────────────────────────────────

class SpeechmaticsVoiceInput:
    """Real-time Speechmatics ASR with tau=0.774s settling buffer.

    Falls back gracefully to text-queue mode if:
    - API key is not set
    - speechmatics-rt is not installed
    - Microphone is not available
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        tau: float = 0.774,
    ) -> None:
        self.api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY", "")
        self.language = language
        self.tau = tau
        self.buffer = TranscriptBuffer(tau=tau)
        self.router = NLPRouter()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_instruction: Optional[Callable[[str], None]] = None
        self._text_queue = TextQueueInput(self._handle_instruction)
        self._use_real_asr = bool(self.api_key)
        self._partial_text: str = ""
        self._settled_text: str = ""

    def _handle_instruction(self, text: str) -> None:
        """Process a settled instruction through the NLP router."""
        if not text.strip():
            return
        self._settled_text = text
        logger.info(f"[Voice] Settled instruction: '{text}'")
        if self._on_instruction:
            self._on_instruction(text)

    def start(self, on_instruction: Callable[[str], None]) -> None:
        """Start listening. Calls on_instruction(text) with each settled utterance."""
        self._on_instruction = on_instruction
        self._running = True

        if self._use_real_asr:
            self._thread = threading.Thread(target=self._run_asr, daemon=True)
            self._thread.start()
        else:
            logger.warning("[Voice] No API key — using text-queue fallback mode")
            self._text_queue.start()

    def send_text(self, text: str) -> None:
        """Inject a text instruction (testing / text fallback mode)."""
        self._text_queue.send(text)
        if not self._text_queue._running:
            self._text_queue.start()

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        self._text_queue.stop()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_partial(self) -> str:
        return self.buffer.get_partial()

    def get_settled(self) -> str:
        return self._settled_text

    def _run_asr(self) -> None:
        """Run real-time ASR via speechmatics-rt in a background thread."""
        try:
            asyncio.run(self._asr_async())
        except Exception as exc:
            logger.error(f"[Voice] ASR thread error: {exc}. Falling back to text queue.")
            self._use_real_asr = False
            self._text_queue.start()

    async def _asr_async(self) -> None:
        """Async real-time ASR using speechmatics-rt SDK."""
        try:
            from speechmatics.client import WebsocketClient
            from speechmatics.models import (
                ConnectionSettings,
                TranscriptionConfig,
                AudioSettings,
            )
        except ImportError:
            # Try older API
            try:
                import speechmatics
                logger.warning("[Voice] Using legacy speechmatics API")
                await self._asr_legacy()
                return
            except Exception as e:
                logger.error(f"[Voice] Cannot import speechmatics: {e}")
                return

        settings = ConnectionSettings(
            url="wss://eu2.rt.speechmatics.com/v2",
            auth_token=self.api_key,
        )
        config = TranscriptionConfig(
            language=self.language,
            operating_point="enhanced",
            enable_partials=True,
        )
        audio_settings = AudioSettings(
            encoding="pcm_f32le",
            sample_rate=16000,
            chunk_size=1024,
        )

        ws = WebsocketClient(settings)

        def on_partial_transcript(msg):
            text = msg.get("metadata", {}).get("transcript", "")
            self.buffer.on_partial(text)
            self._partial_text = text

        def on_final_transcript(msg):
            text = msg.get("metadata", {}).get("transcript", "")
            settled = self.buffer.on_final(text)
            if settled:
                self._handle_instruction(settled)

        def on_end_of_transcript(msg):
            flushed = self.buffer.flush()
            if flushed:
                self._handle_instruction(flushed)

        ws.add_event_handler("AddPartialTranscript", on_partial_transcript)
        ws.add_event_handler("AddTranscript", on_final_transcript)
        ws.add_event_handler("EndOfTranscript", on_end_of_transcript)

        # Try to get audio from microphone
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
            )

            async def audio_generator():
                while self._running:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        yield data
                        await asyncio.sleep(0)
                    except Exception:
                        break

            await ws.run(audio_generator(), config, audio_settings)
            stream.stop_stream()
            stream.close()
            pa.terminate()

        except ImportError:
            logger.warning("[Voice] pyaudio not available — microphone input disabled")
            # Keep thread alive but in text-only mode
            while self._running:
                await asyncio.sleep(0.5)
        except Exception as exc:
            logger.error(f"[Voice] Microphone error: {exc}")
            while self._running:
                await asyncio.sleep(0.5)

    async def _asr_legacy(self) -> None:
        """Legacy ASR fallback using older speechmatics API."""
        logger.warning("[Voice] Legacy speechmatics path — limited functionality")
        while self._running:
            await asyncio.sleep(0.5)


# ─── Verification ─────────────────────────────────────────────────────────────

async def verify_speechmatics_key(api_key: str) -> tuple[bool, str]:
    """Test that the Speechmatics API key is valid by querying the jobs endpoint."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://asr.api.speechmatics.com/v2/jobs",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                return True, "API key valid & authenticated successfully (200 OK)."
            elif r.status_code == 401:
                return False, "Invalid key (401 Unauthorized)"
            else:
                return False, f"Unexpected status {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, f"Connection error: {e}"


if __name__ == "__main__":
    import os

    # Load .env
    env_file = __file__.replace("voice.py", "../.env").replace("brain/../", "")
    from pathlib import Path
    _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("SPEECHMATICS_API_KEY", "")
    print(f"Testing Speechmatics API key ({len(api_key)} chars)...")

    ok, msg = asyncio.run(verify_speechmatics_key(api_key))
    print(f"  {'OK' if ok else 'FAIL'}: {msg}")

    # Test text-queue mode
    print("\nTesting text-queue mode...")
    results = []

    def on_instruction(text):
        results.append(text)
        print(f"  Got instruction: '{text}'")

    voice = SpeechmaticsVoiceInput(api_key=api_key)
    voice.start(on_instruction)
    voice.send_text("set the table")
    time.sleep(0.2)
    voice.send_text("put the plate on the mat")
    time.sleep(0.2)
    voice.stop()

    print(f"\nText-queue test: received {len(results)} instructions")
    router = NLPRouter()
    for r in results:
        cat, cleaned = router.classify(r)
        print(f"  '{r}' -> [{cat}]")

    print("\nDone.")
