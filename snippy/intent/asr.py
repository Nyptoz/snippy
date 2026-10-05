"""Speech recognition, loaded only if the install included the extra.

Recognition is the single heaviest thing Snippy can do, so it is treated as an
optional capability rather than a dependency. Everything here degrades to
"unavailable" and the bot keeps running with slash commands and text triggers.

Transcription runs on a worker thread. ``faster-whisper`` is synchronous and
CPU-bound, and the bot's event loop is the one thing that must never stall while
a voice connection is open.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger("snippy.asr")

VAD_RATE = 16000


class Recognizer:
    """Lazy wrapper around ``faster-whisper``.

    The model is loaded on first use rather than at startup: a guild with spoken
    triggers disabled should not pay for a model it never loads, and enabling
    them should not require a restart.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._model = None
        self._lock = threading.Lock()
        self._loading = False
        self._load_error: str | None = None
        self._semaphore = asyncio.Semaphore(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snippy-asr")

    @property
    def available(self) -> bool:
        """True when recognition is configured and the extra is importable."""
        if not self.config.enabled:
            return False
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def status_line(self) -> str:
        if not self.config.enabled:
            return "disabled"
        if self._load_error:
            return f"unavailable ({self._load_error})"
        if not self.available:
            return "unavailable (install the ASR extra: ./install.sh --asr)"
        return f"ready ({self.config.model})" if self.loaded else f"ready on demand ({self.config.model})"

    def _ensure_model(self):
        """Load the model under a lock so concurrent first calls share one load."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._loading:
                return None
            self._loading = True
        try:
            from faster_whisper import WhisperModel

            log.info("loading ASR model %s on %s", self.config.model, self.config.device)
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._load_error = None
        except Exception as exc:  # pragma: no cover - depends on the host
            self._load_error = str(exc).splitlines()[0][:120]
            log.warning("could not load ASR model: %s", exc)
        finally:
            with self._lock:
                self._loading = False
        return self._model

    def _transcribe_blocking(self, samples: np.ndarray) -> str:
        model = self._ensure_model()
        if model is None:
            return ""
        segments, _info = model.transcribe(
            samples,
            language=self.config.language or None,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio, returning an empty string on failure."""
        if not self.available or samples.size == 0:
            return ""
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    self._pool, self._transcribe_blocking, np.ascontiguousarray(samples, dtype=np.float32)
                )
            except Exception as exc:  # pragma: no cover - depends on the model
                log.warning("transcription failed: %s", exc)
                return ""

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
