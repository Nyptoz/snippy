"""Frame-indexed audio ring buffers.

Every ring in a session shares one timebase: frame index 0 is the moment the
session started listening, and index *n* covers ``[n*20ms, (n+1)*20ms)``. That
is what lets the recorder place each speaker's audio and the mixed audio on the
same clock, so a read covering frames 400-500 returns the same wall-clock
window whether it is slicing the mix or one person's stem.

Two properties matter and are why this is not a plain ``deque``:

* **Gaps become silence.** Discord only sends Opus for users who are actively
  transmitting, so "the last 15 seconds" is mostly not made of packets. Missing
  frames are written as zeros so a clip of a quiet room is 15 real seconds of
  quiet rather than 15 seconds of whatever was last said.
* **Writes may be late or duplicated.** Frames are addressed by index rather
  than appended, so a packet that arrives out of order is dropped and a
  duplicate is ignored instead of corrupting the window.

Reads are clamped to what is still resident, so a request for 120 seconds on a
30-second-old session quietly returns the 30 seconds that exist.
"""

from __future__ import annotations

import threading

import numpy as np

# Discord's voice gateway always sends 20 ms Opus frames at 48 kHz.
FRAME_SECONDS = 0.020
SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_SECONDS)  # 960
CHANNELS = 2

# How far behind the cursor a packet may still land and be folded into the mix.
# Several people talking at once produce their frames for the same 20 ms window
# at slightly different moments, so a strict cursor would drop everyone but the
# first arrival. 150 ms is comfortably wider than the spread and still far
# narrower than the ring itself.
MIX_BACKFILL_FRAMES = 8


def index_for_time(seconds: float) -> int:
    """Convert a session-relative offset into a frame index.

    Rounds rather than truncates: 1.58 / 0.02 is 78.99999999999999 in binary
    floating point, and truncating there would silently drop a frame.
    """
    return int(round(seconds / FRAME_SECONDS))


def time_for_index(index: int) -> float:
    """Convert a frame index back into a session-relative offset."""
    return index * FRAME_SECONDS


class AudioRing:
    """A circular buffer addressed by absolute frame index."""

    def __init__(
        self,
        seconds: float,
        *,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        samples_per_frame: int = SAMPLES_PER_FRAME,
        dtype: np.dtype | type = np.int16,
    ) -> None:
        if seconds <= 0:
            raise ValueError("ring must hold at least one frame")
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_frame = samples_per_frame
        self.capacity = max(1, int(round(seconds / FRAME_SECONDS)))
        self._data = np.zeros(
            (self.capacity, samples_per_frame, channels), dtype=dtype
        )
        self._next = 0
        self._lock = threading.Lock()
        self._mix_backfill = MIX_BACKFILL_FRAMES
        self.gap_frames = 0
        self.dropped_frames = 0
        self.written_frames = 0

    # -- state ---------------------------------------------------------------

    @property
    def oldest_index(self) -> int:
        return max(0, self._next - self.capacity)

    @property
    def newest_index(self) -> int:
        return self._next

    def available_seconds(self) -> float:
        return (self._next - self.oldest_index) * FRAME_SECONDS

    def clear(self) -> None:
        with self._lock:
            self._data[:] = 0
            self._next = 0

    # -- writing -------------------------------------------------------------

    def _prepare(self, index: int, backfill: int = 0) -> tuple[int, bool] | None:
        """Resolve an index to a slot, zeroing any frames skipped on the way.

        Returns ``(slot, advanced)``, or None if the frame is too old to still
        be part of the mix. ``backfill`` lets a frame that is still filling
        accept more contributors; without it a second speaker's packet for the
        same 20 ms frame would be discarded as stale and the mix would only
        ever contain whoever spoke first. Caller must hold the lock.
        """
        if index < max(self.oldest_index, self._next - backfill):
            # A late or duplicated packet. Discord does not retransmit, so the
            # frame we already have is as good as this one.
            self.dropped_frames += 1
            return None
        if index < self._next:
            return index % self.capacity, False
        gap = index - self._next
        if gap >= self.capacity:
            # We fell further behind than the whole window, so everything we
            # hold is stale. Wipe it rather than zeroing frame by frame.
            self._data[:] = 0
            self.gap_frames += self.capacity
            self._next = index
            return index % self.capacity, True
        if gap:
            self._zero_range(self._next, gap)
            self.gap_frames += gap
            self._next = index
        return index % self.capacity, True

    def _zero_range(self, start: int, count: int) -> None:
        first = min(count, self.capacity - (start % self.capacity))
        self._data[start % self.capacity : start % self.capacity + first] = 0
        rest = count - first
        if rest:
            self._data[:rest] = 0

    def write(self, index: int, frame: np.ndarray) -> bool:
        """Store one frame of audio at an absolute index."""
        with self._lock:
            prepared = self._prepare(int(index))
            if prepared is None:
                return False
            slot, advanced = prepared
            self._data[slot] = frame
            if advanced:
                self._next = int(index) + 1
                self.written_frames += 1
            return True

    def accumulate(self, index: int, frame: np.ndarray) -> bool:
        """Add one frame into the buffer, saturating rather than wrapping.

        Used to build the mix: summing ten loud speakers must clip, not wrap
        around into a burst of noise. The addition is done in int32 because
        numpy would otherwise overflow the int16 destination *before* the
        clamp ever saw the value.
        """
        with self._lock:
            prepared = self._prepare(int(index), backfill=self._mix_backfill)
            if prepared is None:
                return False
            slot, advanced = prepared
            total = self._data[slot].astype(np.int32) + frame.astype(np.int32)
            self._data[slot] = _saturate(total, self._data.dtype)
            if advanced:
                self._next = int(index) + 1
                self.written_frames += 1
            return True

    # -- reading -------------------------------------------------------------

    def read(self, start: float, count_frames: int) -> np.ndarray:
        """Read `count_frames` frames from a session-relative start time.

        Returns a ``(frames, samples_per_frame, channels)`` view, or fewer
        frames when the requested range is only partly resident.
        """
        with self._lock:
            begin = max(int(index_for_time(start)), self.oldest_index)
            count = min(int(count_frames), self._next - begin)
            if count <= 0:
                return np.zeros((0, self.samples_per_frame, self.channels), dtype=self._data.dtype)
            slots = (begin + np.arange(count)) % self.capacity
            return self._data[slots].copy()

    def read_seconds(self, end_time: float, length: float) -> np.ndarray:
        """Read `length` seconds ending at session-relative `end_time`."""
        start = max(0.0, end_time - length)
        return self.read(start, int(round(length / FRAME_SECONDS)))

    def to_pcm(self, frames: np.ndarray) -> np.ndarray:
        """Flatten a block of frames into interleaved samples for ffmpeg."""
        if frames.size == 0:
            return np.zeros((0, self.channels), dtype=self._data.dtype)
        return frames.reshape(-1, self.channels)


def _saturate(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Clamp into `dtype`'s range. The clamp must target the *destination*,
    not the array being stored, or the subsequent cast wraps instead."""
    info = np.iinfo(np.dtype(dtype))
    return np.clip(values, info.min, info.max).astype(dtype, copy=False)


class RingSet:
    """The mix ring plus a bounded set of per-speaker stems sharing a timebase.

    Stems are allocated lazily on first speech, which matters when a call has
    thirty participants but only two of them ever talk: an unallocated stem
    costs nothing. Once the cap is reached, the least recently active speaker's
    stem is recycled, and callers fall back to the mix for anyone evicted.
    """

    def __init__(self, cfg, *, mix_dtype: np.dtype | type = np.int16) -> None:
        self.mix = AudioRing(cfg.mix_seconds, dtype=mix_dtype)
        self.stem_seconds = cfg.stem_seconds
        self.max_stem_users = cfg.max_stem_users
        self._stems: dict[int, AudioRing] = {}
        self._last_seen: dict[int, int] = {}
        # A plain monotonic counter, not the mix cursor: reading a stem does
        # not advance the mix, so a cursor-based "last seen" would never change
        # and eviction would always fall on the same unlucky speaker.
        self._clock = 0
        self._lock = threading.Lock()
        itemsize = np.dtype(mix_dtype).itemsize
        self._stem_bytes = (
            int(round(cfg.stem_seconds / FRAME_SECONDS))
            * SAMPLES_PER_FRAME
            * CHANNELS
            * itemsize
        )

    def stem(self, user_id: int) -> AudioRing | None:
        """Return this speaker's stem ring, or None if they have been evicted."""
        with self._lock:
            ring = self._stems.get(user_id)
            if ring is not None:
                self._clock += 1
                self._last_seen[user_id] = self._clock
            return ring

    def stem_for_write(self, user_id: int) -> AudioRing:
        """Return this speaker's stem ring, allocating or evicting if needed."""
        with self._lock:
            ring = self._stems.get(user_id)
            if ring is None:
                if len(self._stems) >= self.max_stem_users:
                    victim = min(self._last_seen, key=self._last_seen.get)
                    del self._stems[victim]
                    self._last_seen.pop(victim, None)
                ring = AudioRing(self.stem_seconds)
                self._stems[user_id] = ring
            self._clock += 1
            self._last_seen[user_id] = self._clock
            return ring

    def known_users(self) -> list[int]:
        with self._lock:
            return list(self._stems)

    def clear(self) -> None:
        self.mix.clear()
        with self._lock:
            self._stems.clear()
            self._last_seen.clear()

    def memory_bytes(self) -> int:
        with self._lock:
            stems = len(self._stems)
        return int(self.mix._data.nbytes) + stems * self._stem_bytes
