"""Voice activity detection and resampling.

Snippy needs to know two things about the audio it is holding: where speech
starts and stops, and how loud each moment was. The first drives silence
trimming and the ``active`` clip style; the second drives the waveform in
``/when`` and the loudest-moment suggestion.

The detector is deliberately cheap. It runs on a 16 kHz mono shadow of the
audio, and it is thresholded against a rolling noise floor rather than a fixed
number, because a call in a quiet room and a call next to a washing machine
need different thresholds. Hysteresis and a hangover stop the classic failure
mode where one person saying "the" gets chopped into three utterances.
"""

from __future__ import annotations

import numpy as np

VAD_RATE = 16000
FRAME_SECONDS = 0.020
INT16_MAX = 32768.0

# About -42 dBFS, a level well below speech but above true digital silence.
_ABSOLUTE_FLOOR = 10 ** (-42 / 20)


def to_float(pcm: np.ndarray) -> np.ndarray:
    """Convert an int16 sample block to float in -1.0 to 1.0.

    Every level measurement, threshold, and speech-to-text call in Snippy works
    in this scale. Skipping the conversion is silent and catastrophic: dB
    thresholds stop meaning anything and the transcriber is handed audio
    thousands of times too loud.
    """
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    return pcm.astype(np.float32) / INT16_MAX


def resample_mono(pcm: np.ndarray, source_rate: int, target_rate: int = VAD_RATE) -> np.ndarray:
    """Downmix to mono, normalise, and resample with a windowed-sinc FIR.

    Discord's voice is 48 kHz stereo, so this is always a 3:1 decimation to
    16 kHz. A plain reshape-and-mean would alias high frequencies down into the
    speech band and make the level readings wrong, so the signal is lowpassed
    at the new Nyquist frequency first.
    """
    if pcm.size == 0:
        return np.zeros((0,), dtype=np.float32)
    mono = to_float(pcm)
    mono = mono.mean(axis=1) if mono.ndim > 1 else mono
    if source_rate == target_rate:
        return mono
    if source_rate < target_rate:
        return _upsample(mono, source_rate, target_rate)

    taps = _lowpass_taps(source_rate, target_rate)
    filtered = np.convolve(mono, taps, mode="same")
    step = source_rate / target_rate
    usable = int((len(filtered) - 1) / step) + 1
    return np.ascontiguousarray(filtered[:: int(step)][:usable], dtype=np.float32)


def _lowpass_taps(source_rate: int, target_rate: int, taps: int = 33) -> np.ndarray:
    """Windowed-sinc lowpass at the target Nyquist frequency."""
    cutoff = min(0.5, target_rate / (2.0 * source_rate))
    n = np.arange(taps) - (taps - 1) / 2.0
    sinc = np.sinc(2.0 * cutoff * n)
    window = np.hamming(taps)
    kernel = sinc * window
    return (kernel / kernel.sum()).astype(np.float32)


def _upsample(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear upsample, used only if a non-Discord source rate shows up."""
    step = target_rate / source_rate
    positions = np.arange(int(len(mono) * step)) / step
    index = np.clip(positions.astype(np.int64), 0, len(mono) - 1)
    return mono[index].astype(np.float32)


def frame_rms(mono: np.ndarray, frame_samples: int = VAD_RATE // 50) -> np.ndarray:
    """Root-mean-square level per 20 ms frame, as a normalized 0..1 value."""
    if mono.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if np.max(np.abs(mono)) > 1.5:
        # A caller handed us raw int16 samples. Scale them rather than report
        # a nonsense level that would read as permanently clipping.
        mono = mono / INT16_MAX
    usable = (len(mono) // frame_samples) * frame_samples
    if usable == 0:
        return np.array([float(np.sqrt(np.mean(np.square(mono))))], dtype=np.float32)
    frames = mono[:usable].reshape(-1, frame_samples)
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)


def to_db(level: np.ndarray | float) -> np.ndarray | float:
    """Convert a linear amplitude to dBFS, floored so silence stays finite."""
    return 20.0 * np.log10(np.maximum(level, 1e-9))


class ActivityDetector:
    """Adaptive speech detector over a rolling level track.

    Feed it 20 ms levels in order; it returns speech decisions for the frames
    it has now seen. The noise floor follows the quiet percentile of recent
    history with slow upward drift, so a door closing does not permanently
    desensitise the detector for the rest of the session.
    """

    def __init__(
        self,
        *,
        on_margin_db: float = 9.0,
        off_margin_db: float = 5.0,
        hangover_frames: int = 12,
        history_frames: int = 2500,
    ) -> None:
        self.on_margin_db = on_margin_db
        self.off_margin_db = off_margin_db
        self.hangover_frames = hangover_frames
        self.history_frames = history_frames
        self._levels = np.zeros(0, dtype=np.float32)
        self._floor_db = -60.0
        self._speaking = False
        self._counter = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def push(self, level: float) -> bool:
        """Add one frame's level and return whether it counts as speech."""
        self._levels = np.append(self._levels, np.float32(level))
        if len(self._levels) > self.history_frames:
            self._levels = self._levels[-self.history_frames :]
            if len(self._levels) > 200:
                noise = float(np.percentile(self._levels, 15))
                noise_db = float(to_db(noise))
                # Rise quickly so a sudden loud noise does not raise the floor,
                # fall slowly so a real noise floor is not forgotten.
                if noise_db > self._floor_db:
                    self._floor_db += (noise_db - self._floor_db) * 0.5
                else:
                    self._floor_db += (noise_db - self._floor_db) * 0.01
                self._floor_db = min(self._floor_db, -25.0)
                self._floor_db = max(self._floor_db, -75.0)

        level_db = float(to_db(level))
        effective_floor = max(self._floor_db, float(to_db(_ABSOLUTE_FLOOR)))
        if self._speaking:
            if level_db > effective_floor + self.off_margin_db:
                self._counter = self.hangover_frames
            else:
                self._counter -= 1
                if self._counter <= 0:
                    self._speaking = False
        else:
            if level_db > effective_floor + self.on_margin_db:
                self._speaking = True
                self._counter = self.hangover_frames
        return self._speaking

    def reset(self) -> None:
        self._levels = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._counter = 0


def speech_runs(flags: np.ndarray, *, min_frames: int = 12, merge_gap: int = 10) -> list[tuple[int, int]]:
    """Collapse a per-frame speech track into ``(start, end)`` index pairs.

    Short blips are dropped and short gaps are bridged, so a laugh or a
    "mm-hmm" survives as one run instead of a scatter of frames.
    """
    if flags.size == 0:
        return []
    merged = flags.copy()
    for gap in _gaps(flags, max_gap=merge_gap):
        merged[gap[0] : gap[1] + 1] = True
    runs = []
    start: int | None = None
    for index, active in enumerate(merged):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= min_frames:
                runs.append((start, index))
            start = None
    if start is not None and len(merged) - start >= min_frames:
        runs.append((start, len(merged)))
    return runs


def _gaps(flags: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` index pairs of short *interior* False runs.

    Leading and trailing silence are deliberately excluded: merging them would
    swallow the quiet head of the track and stretch the first run backwards
    from where the speech actually began.
    """
    if flags.size == 0 or max_gap <= 0:
        return []
    padded = np.concatenate(([True], flags, [True]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == -1)
    ends = np.flatnonzero(transitions == 1)
    last = len(flags) - 1
    return [
        (int(s), int(e) - 1)
        for s, e in zip(starts, ends)
        if e - s <= max_gap and s > 0 and (e - 1) < last
    ]


def trim_silence(
    pcm: np.ndarray,
    *,
    sample_rate: int = 48000,
    threshold_db: float = -45.0,
    keep_ms: int = 60,
) -> np.ndarray:
    """Strip leading and trailing silence, keeping a short natural-sounding pad.

    Interior pauses are left alone. Cutting every pause turns a conversation
    into a ransom note; the ``active`` style exists for people who want that.
    """
    if pcm.size == 0:
        return pcm
    mono = to_float(pcm)
    mono = mono.mean(axis=1) if mono.ndim > 1 else mono
    if mono.size == 0:
        return pcm
    block = max(1, sample_rate // 100)
    usable = (len(mono) // block) * block
    if usable == 0:
        return pcm
    levels = np.sqrt(np.mean(np.square(mono[:usable].reshape(-1, block)), axis=1))
    loud = np.flatnonzero(to_db(levels) > threshold_db)
    if loud.size == 0:
        return pcm
    pad = max(1, int(keep_ms / 1000.0 * sample_rate / block))
    start = max(0, int(loud[0]) - pad) * block
    end = min(len(mono), (int(loud[-1]) + 1 + pad) * block)
    return pcm[start:end]


def squeeze(
    pcm: np.ndarray,
    *,
    sample_rate: int = 48000,
    threshold_db: float | None = None,
    min_gap_ms: int = 250,
    min_run_ms: int = 120,
    join_ms: int = 70,
) -> np.ndarray:
    """Cut the dead air out of a clip, keeping the conversation intact.

    Backs the ``active`` style. Gaps shorter than `min_gap_ms` survive, because
    a beat of silence between two lines is part of how the line lands; only
    genuine dead air is removed. Each splice gets a short fade, so joining at a
    threshold crossing does not click.
    """
    if pcm.size == 0:
        return pcm
    block = max(1, sample_rate // 100)
    usable = (len(pcm) // block) * block
    if usable == 0:
        return pcm
    levels = np.sqrt(
        np.mean(np.square(to_float(pcm[:usable]).reshape(-1, block)), axis=1)
    )
    if threshold_db is None:
        # Relative to the clip itself, so this works in a quiet room and a loud
        # one alike. Falls back to an absolute floor on near-silent input.
        noise_db = float(np.percentile(to_db(levels), 20))
        threshold_db = max(-45.0, noise_db + 8.0)
    flags = to_db(levels) > threshold_db
    runs = speech_runs(flags, min_frames=int(min_run_ms / 10), merge_gap=int(min_gap_ms / 10))
    if not runs:
        return pcm
    if len(runs) == 1:
        return pcm[runs[0][0] * block : runs[0][1] * block]

    fade = max(1, int(sample_rate * join_ms / 1000.0 / 2))
    pieces: list[np.ndarray] = []
    for index, (start, end) in enumerate(runs):
        block_pcm = pcm[start * block : end * block].copy()
        if index > 0:
            head = min(fade, len(block_pcm))
            if head:
                block_pcm[:head] = _fade(block_pcm[:head], np.linspace(0, 1, head))
        if index < len(runs) - 1:
            tail = min(fade, len(block_pcm))
            if tail:
                block_pcm[-tail:] = _fade(block_pcm[-tail:], np.linspace(1, 0, tail))
        pieces.append(block_pcm)
    return np.concatenate(pieces, axis=0)


def _fade(block: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights.reshape(-1, 1) if block.ndim > 1 else weights
    return (block * weights).astype(block.dtype)


def peak_envelope(pcm: np.ndarray, buckets: int = 180) -> np.ndarray:
    """Downsample a clip to a fixed number of peak values for drawing."""
    if pcm.size == 0 or buckets <= 0:
        return np.zeros(0, dtype=np.float32)
    mono = np.abs(pcm.mean(axis=1) if pcm.ndim > 1 else pcm)
    if mono.size == 0:
        return np.zeros(buckets, dtype=np.float32)
    edges = np.linspace(0, len(mono), buckets + 1).astype(int)
    peaks = [float(mono[a:b].max()) if b > a else 0.0 for a, b in zip(edges, edges[1:])]
    highest = max(peaks) or 1.0
    return (np.array(peaks, dtype=np.float32) / highest).clip(0.0, 1.0)
