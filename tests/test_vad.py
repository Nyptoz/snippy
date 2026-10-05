"""Voice activity detection, resampling, and silence handling."""

from __future__ import annotations

import numpy as np
import pytest

from snippy.voice import vad
from snippy.voice.rings import FRAME_SECONDS

RATE = 16000


def test_resample_downmixes_and_halves_the_rate():
    pcm = (np.random.default_rng(1).standard_normal((4800, 2)) * 1000).astype(np.int16)
    out = vad.resample_mono(pcm, 48000)
    assert out.dtype == np.float32
    assert abs(len(out) - 1600) <= 2


def test_resample_normalises_to_the_float_scale():
    """Every level and threshold downstream assumes -1.0 to 1.0."""
    t = np.arange(960) / 48000.0
    full_scale = (np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    pcm = np.stack([full_scale, full_scale], axis=1)
    out = vad.resample_mono(pcm, 48000)
    assert np.max(np.abs(out)) == pytest.approx(1.0, abs=0.02)


def test_resample_handles_empty_input():
    assert vad.resample_mono(np.zeros((0, 2), dtype=np.int16), 48000).size == 0


def test_frame_rms_scales_raw_samples_rather_than_reporting_nonsense():
    quiet = np.full((320,), 3000.0, dtype=np.float32)
    loud = np.full((320,), 30000.0, dtype=np.float32)
    assert vad.frame_rms(loud)[0] == pytest.approx(vad.frame_rms(quiet)[0] * 10, rel=0.01)


def test_detector_finds_two_utterances_in_silence():
    t = np.arange(RATE * 3) / RATE
    signal = np.zeros(RATE * 3, dtype=np.float32)
    rng = np.random.default_rng(0)
    for start, end in ((0.3, 0.9), (1.5, 2.4)):
        mask = (t >= start) & (t < end)
        signal[mask] = 0.3 * np.sin(2 * np.pi * 220 * t[mask])
        signal[mask] += 0.01 * rng.standard_normal(int(mask.sum()))

    detector = vad.ActivityDetector()
    flags = np.array([detector.push(float(level)) for level in vad.frame_rms(signal)])
    runs = vad.speech_runs(flags)

    assert len(runs) == 2
    starts = [start * FRAME_SECONDS for start, _ in runs]
    ends = [end * FRAME_SECONDS for _, end in runs]
    # Starts track the real onsets; the hangover deliberately runs each run
    # past the last voiced sample, so the ends are only loosely bounded.
    assert starts[0] == pytest.approx(0.3, abs=0.1)
    assert starts[1] == pytest.approx(1.5, abs=0.1)
    assert ends[0] > 0.85
    assert ends[1] > 2.35


def test_detector_ignores_a_constant_noise_floor():
    noise = np.random.default_rng(2).standard_normal(RATE * 2).astype(np.float32) * 0.02
    detector = vad.ActivityDetector()
    flags = [detector.push(float(level)) for level in vad.frame_rms(noise)]
    assert not any(flags)


def test_short_blips_are_dropped_and_short_gaps_are_bridged():
    blip = np.zeros(200, dtype=bool)
    blip[10:20] = True
    assert vad.speech_runs(blip) == []

    pair = np.zeros(200, dtype=bool)
    pair[10:25] = True
    pair[29:40] = True
    assert vad.speech_runs(pair) == [(10, 40)]


def test_widely_separated_runs_stay_separate():
    flags = np.zeros(200, dtype=bool)
    flags[10:25] = True
    flags[120:140] = True
    assert vad.speech_runs(flags) == [(10, 25), (120, 140)]


def test_trim_silence_removes_the_edges_only():
    body = np.full((480, 2), 5000, dtype=np.int16)
    pcm = np.concatenate([np.zeros((480 * 20, 2), np.int16), body, np.zeros((480 * 30, 2), np.int16)])
    trimmed = vad.trim_silence(pcm, keep_ms=0)
    assert len(trimmed) < len(pcm)
    assert np.abs(trimmed).max() == 5000


def test_trim_silence_returns_silence_untouched():
    pcm = np.zeros((4800, 2), dtype=np.int16)
    assert vad.trim_silence(pcm) is pcm


def test_squeeze_cuts_interior_dead_air():
    loud = np.full((480 * 10, 2), 6000, dtype=np.int16)
    pcm = np.concatenate(
        [
            np.zeros((480 * 10, 2), np.int16),
            loud,
            np.zeros((480 * 80, 2), np.int16),
            loud,
            np.zeros((480 * 10, 2), np.int16),
        ]
    )
    squeezed = vad.squeeze(pcm, min_gap_ms=300, min_run_ms=100, join_ms=0)
    assert 0 < len(squeezed) < len(pcm)
    # Two ten-block bursts plus a little, not the original with its holes.
    assert len(squeezed) == pytest.approx(2 * 480 * 10, rel=0.25)


def test_squeeze_leaves_a_clip_without_gaps_alone():
    pcm = np.full((480 * 30, 2), 6000, dtype=np.int16)
    assert len(vad.squeeze(pcm)) == len(pcm)


def test_peak_envelope_is_normalised():
    pcm = (np.random.default_rng(4).standard_normal((48000, 2)) * 8000).astype(np.int16)
    envelope = vad.peak_envelope(pcm, buckets=120)
    assert envelope.shape == (120,)
    assert envelope.max() == pytest.approx(1.0, abs=1e-5)
    assert (envelope >= 0).all() and (envelope <= 1).all()
