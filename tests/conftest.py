"""Shared test fixtures.

The suite runs entirely offline: no Discord connection, and no network. Audio is
synthesised, ffmpeg is invoked directly, and anything needing a bot is stubbed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from snippy.config import GuildConfig, RingConfig
from snippy.voice.recorder import SnippySink

RATE = 48000
FRAME = 960


def speech(amplitude: float, seconds: float = 0.02, freq: float = 220.0) -> np.ndarray:
    """One 20 ms stereo frame of a steady tone at a given amplitude."""
    t = np.arange(int(RATE * seconds)) / RATE
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return np.stack([mono, mono], axis=1)


def tone(seconds: float, amplitude: float = 9000, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return np.stack([mono, mono], axis=1)


class FakeVoiceData:
    def __init__(self, pcm):
        self.pcm = pcm


class FakeUser:
    def __init__(self, user_id: int):
        self.id = user_id


class FakeClock:
    """A manually advanced clock, so capture tests need no real sleeping.

    Frame indices in the rings are derived from elapsed time, so feeding audio
    faster than real time would otherwise pile every frame onto one index.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ring_config() -> RingConfig:
    return RingConfig(mix_seconds=10.0, stem_seconds=5.0, vad_seconds=30.0, max_stem_users=4)


@pytest.fixture
def config() -> GuildConfig:
    return GuildConfig()


@pytest.fixture
def sink(ring_config: RingConfig, clock: FakeClock) -> SnippySink:
    return SnippySink(
        ring_config=ring_config,
        guild_id=1,
        ignored_users=lambda: set(),
        min_utterance_ms=100,
        clock=clock,
        wall_clock=clock,
    )


def feed(sink: SnippySink, clock: FakeClock, frames, user_id: int) -> None:
    """Push one frame per 20 ms of fake time."""
    sink.write(FakeUser(user_id), FakeVoiceData(frames))
    clock.advance(0.02)


@pytest.fixture
def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


@pytest.fixture
def requires_ffmpeg(ffmpeg):
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    return ffmpeg


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260928)
