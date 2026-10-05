"""The capture sink.

This is the one place that touches every incoming Opus frame. It demuxes
Discord's per-user voice streams, decodes each one exactly once, and fans the
result out to the places that need it: the per-speaker stem ring, the mixed
ring, the low-rate activity track, and the archive.

Everything here is synchronous and runs on the event loop, because that is how
``discord-ext-voice-recv`` calls the sink. The rule that follows from that is
that this function must not block: no disk, no subprocess, no network. The
archive is fed by a separate pump task that reads the finished mix, so writing
a clip or encoding to disk can never add latency to the voice socket.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import vad
from .archive import SpeechRunTracker
from .rings import CHANNELS, FRAME_SECONDS, SAMPLE_RATE, SAMPLES_PER_FRAME, AudioRing, RingSet, index_for_time

log = logging.getLogger("snippy.recorder")

try:  # PyAV ships with discord.py, but degrade rather than crash without it.
    import av

    HAVE_AV = True
except Exception:  # pragma: no cover
    av = None
    HAVE_AV = False


class OpusDecoder:
    """A persistent Opus decoder for one voice stream.

    A fresh ``CodecContext`` per packet would mean re-initialising the decoder
    fifty times a second per speaker. Keeping it alive also lets libopus carry
    its internal state across frames, which is what keeps long streams free of
    clicks at the seams.
    """

    def __init__(self, rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        if not HAVE_AV:
            raise RuntimeError("PyAV is unavailable, cannot decode Opus")
        self._context = av.CodecContext.create("opus", "r")
        # Resampling to a known packed format means downstream code never has to
        # care whether libopus produced planar or interleaved samples.
        self._resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="stereo", rate=rate
        )
        self.rate = rate
        self.channels = channels

    def decode(self, packet: bytes) -> np.ndarray | None:
        """Decode one Opus packet into ``(samples, channels)`` int16."""
        try:
            frames = self._context.decode(av.Packet(bytes(packet)))
        except Exception as exc:  # pragma: no cover - malformed packet
            log.debug("opus decode failed: %s", exc)
            return None
        if not frames:
            return None
        chunks = [
            resampled.to_ndarray()
            for resampled in self._resampler.resample(frames)
            if resampled.samples
        ]
        if not chunks:
            return None
        block = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        samples = block.size // self.channels
        if samples == 0:
            return None
        return np.ascontiguousarray(
            block.reshape(-1)[: samples * self.channels].astype(np.int16, copy=False)
        ).reshape(-1, self.channels)


class DecoderPool:
    """Bounded cache of decoders, one per user stream."""

    def __init__(self, limit: int = 64) -> None:
        self._decoders: dict[int, OpusDecoder] = {}
        self._limit = limit

    def get(self, user_id: int) -> OpusDecoder:
        decoder = self._decoders.get(user_id)
        if decoder is not None:
            return decoder
        if len(self._decoders) >= self._limit:
            self._decoders.pop(next(iter(self._decoders)))
        decoder = OpusDecoder()
        self._decoders[user_id] = decoder
        return decoder

    def drop(self, user_id: int) -> None:
        self._decoders.pop(user_id, None)

    def clear(self) -> None:
        self._decoders.clear()


@dataclass
class Utterance:
    """A completed stretch of speech, in session-relative seconds."""

    start: float
    end: float
    speaker: int | None = None
    text: str | None = None


class SnippySink:
    """Fan audio out to the rings, the activity track, and the run tracker.

    Implements the ``AudioSink`` interface rather than subclassing it, so the
    bot keeps working even if the extension changes its base class signature.
    """

    def __init__(
        self,
        *,
        ring_config,
        guild_id: int,
        ignored_users: Callable[[], set[int]],
        min_utterance_ms: int = 400,
        on_packet=None,
        clock=time.monotonic,
        wall_clock=time.time,
    ) -> None:
        self.ring_config = ring_config
        self.guild_id = guild_id
        self._ignored_users = ignored_users
        self._min_utterance_ms = min_utterance_ms
        self._on_packet = on_packet
        # Frame indices come from elapsed time, so the clock is injectable:
        # tests advance a fake one instead of sleeping through real seconds.
        self._clock = clock
        self._wall_clock = wall_clock

        self.rings = RingSet(ring_config)
        self.vad_ring = AudioRing(
            ring_config.vad_seconds,
            sample_rate=vad.VAD_RATE,
            channels=1,
            samples_per_frame=vad.VAD_RATE // 50,
        )
        self.detector = vad.ActivityDetector()
        self.tracker = SpeechRunTracker()
        self.decoders = DecoderPool()
        self.utterances: deque[Utterance] = deque(maxlen=64)

        self.started_at = self._clock()
        self.started_wall = self._wall_clock()
        self.packets = 0
        self.decode_failures = 0
        self.ignored_packets = 0
        self._utterance_start: float | None = None
        self._utterance_speaker: int | None = None
        self._levels = np.zeros(0, dtype=np.float32)

    # -- AudioSink interface -------------------------------------------------

    def wants_opus(self) -> bool:
        # Always true: the archive is fed straight from Discord's own packets
        # so nothing is re-encoded on the way to disk.
        return True

    def write(self, user, data) -> None:
        user_id = getattr(user, "id", None)
        if user_id is None:
            return
        if user_id in self._ignored_users():
            self.ignored_packets += 1
            return

        index = self._current_index()
        pcm = self._decode(user_id, data)
        if pcm is None or pcm.size == 0:
            return

        self.rings.stem_for_write(user_id).write(index, pcm)
        self.rings.mix.accumulate(index, pcm)
        self._track_activity(user_id, index, pcm)
        self.packets += 1
        if self._on_packet is not None:
            self._on_packet(user_id, index)

    def cleanup(self) -> None:
        self.decoders.clear()

    # -- internals -----------------------------------------------------------

    def _current_index(self) -> int:
        return index_for_time(self._clock() - self.started_at)

    def _decode(self, user_id: int, data) -> np.ndarray | None:
        pcm = getattr(data, "pcm", None)
        if pcm is not None:
            block = np.asarray(pcm, dtype=np.int16)
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            if block.shape[1] != CHANNELS and block.shape[1] == 1:
                block = np.repeat(block, CHANNELS, axis=1)
            return block
        packet = getattr(data, "data", None)
        if not packet:
            return None
        try:
            decoded = self.decoders.get(user_id).decode(packet)
        except RuntimeError:
            self.decode_failures += 1
            return None
        if decoded is None:
            self.decode_failures += 1
        return decoded

    def _track_activity(self, user_id: int, index: int, pcm: np.ndarray) -> None:
        """Update the low-rate activity track, utterances, and run timeline."""
        now = self._wall_clock()
        mono = vad.resample_mono(pcm, SAMPLE_RATE)
        if mono.size == 0:
            return
        level = float(np.sqrt(np.mean(np.square(mono))))
        speaking = self.detector.push(level)
        self._levels = np.append(self._levels, np.float32(level))
        if len(self._levels) > 4096:
            self._levels = self._levels[-4096:]

        # The mixed track drives utterance boundaries; the per-user feed drives
        # the speaker timeline that /when draws. A silent channel closes every
        # open run, because a quiet speaker emits no packets to close their own.
        if speaking:
            self.tracker.note(user_id, now, True)
        else:
            self.tracker.silence(now)
        self._note_utterance(speaking, user_id)

        padded = _fit(mono, vad.VAD_RATE // 50)
        self.vad_ring.write(index, padded.reshape(-1, 1))

    def _note_utterance(self, speaking: bool, user_id: int) -> None:
        elapsed = self._clock() - self.started_at
        if speaking:
            if self._utterance_start is None:
                self._utterance_start = max(0.0, elapsed - FRAME_SECONDS)
                self._utterance_speaker = user_id
            return
        if self._utterance_start is None:
            return
        start = self._utterance_start
        self._utterance_start = None
        duration = elapsed - start
        if duration * 1000 >= self._min_utterance_ms:
            self.utterances.append(
                Utterance(start=start, end=elapsed, speaker=self._utterance_speaker)
            )
        self._utterance_speaker = None

    # -- reads ---------------------------------------------------------------

    def elapsed(self) -> float:
        return self._clock() - self.started_at

    def read_mix(self, start: float, duration: float) -> np.ndarray:
        frames = self.rings.mix.read(start, int(round(duration / FRAME_SECONDS)))
        return self.rings.mix.to_pcm(frames)

    def read_stem(self, user_id: int, start: float, duration: float) -> np.ndarray | None:
        ring = self.rings.stem(user_id)
        if ring is None:
            return None
        frames = ring.read(start, int(round(duration / FRAME_SECONDS)))
        if frames.size == 0:
            return None
        return ring.to_pcm(frames)

    def read_activity(self, start: float, duration: float) -> np.ndarray:
        """16 kHz mono for the same window, for transcription."""
        count = int(round(duration / FRAME_SECONDS))
        frames = self.vad_ring.read(start, count)
        if frames.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return frames.reshape(-1).astype(np.float32)

    def levels_between(self, start: float, duration: float) -> np.ndarray:
        """Recent per-frame levels, for peak and loudest-moment detection."""
        count = int(round(duration / FRAME_SECONDS))
        if count <= 0 or not len(self._levels):
            return np.zeros((0,), dtype=np.float32)
        return self._levels[-count:]

    def drain_utterances(self) -> list[Utterance]:
        items = list(self.utterances)
        self.utterances.clear()
        return items

    def reset(self) -> None:
        """Clear the rings while keeping the decoders, used after a reconnect."""
        self.rings.clear()
        self.vad_ring.clear()
        self.detector.reset()
        self.utterances.clear()
        self._utterance_start = None
        self._levels = np.zeros(0, dtype=np.float32)
        self.started_at = self._clock()
        self.started_wall = self._wall_clock()
        self.tracker.reset(self.started_wall)


def _fit(mono: np.ndarray, size: int) -> np.ndarray:
    """Force a mono block to exactly `size` samples, padding or trimming.

    Resampling 48 kHz to 16 kHz is exactly 3:1 so this is normally a no-op, but
    a decoder that emitted a short frame would otherwise shift every later
    sample in the activity track.
    """
    if len(mono) == size:
        return mono
    if len(mono) > size:
        return mono[:size]
    return np.concatenate([mono, np.zeros(size - len(mono), dtype=mono.dtype)])
