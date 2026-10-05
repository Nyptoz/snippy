"""Clip rendering, end to end through ffmpeg with synthetic audio.

These exercise the real encoder rather than a mock, because the interesting
failures in a clipper (a window of pure silence, a mix that never sums, a trim
that cuts the wrong edge) are exactly the ones a mock would hide.

Audio is fed against a fake clock, so the ring's frame indices advance the way
they would in a real call without the test waiting four real seconds.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from snippy.clips import ClipBuilder, ClipRequest, is_silent

from .conftest import FakeClock, FakeUser, FakeVoiceData, feed, speech

pytestmark = pytest.mark.usefixtures("requires_ffmpeg")

# Two speakers taking turns in four one-second blocks: Ken speaks in blocks 0
# and 2, Marisol in 1 and 3, and the third second of each pair is quiet.
BLOCKS = [(5, 9000), (9, 5000), (5, 9000), (9, 5000)]
FRAMES_PER_BLOCK = 50
QUIET_FRAMES = 20


@pytest.fixture
def recorded(sink, clock):
    """Four seconds of two speakers, with genuine silences between turns."""
    for user_id, amplitude in BLOCKS:
        for _ in range(FRAMES_PER_BLOCK):
            feed(sink, clock, speech(amplitude), user_id)
        for _ in range(QUIET_FRAMES):
            feed(sink, clock, speech(0), user_id)
    return sink


@pytest.fixture
def builder(config, recorded, tmp_path):
    instance = ClipBuilder(tmp_path, config, db=None)
    instance.sink_provider = lambda guild_id, channel_id: recorded
    return instance


def request_for(sink, **kwargs) -> ClipRequest:
    elapsed = sink.elapsed()
    base = dict(
        guild_id=1,
        channel_id=100,
        session_id=None,
        start=max(0.0, elapsed - 3.0),
        end=elapsed,
        source="ram",
        requester_id=42,
    )
    base.update(kwargs)
    return ClipRequest(**base)


async def test_mix_renders_and_is_playable(builder, recorded):
    result = await builder.build(request_for(recorded))
    try:
        assert result.path.exists()
        assert result.path.stat().st_size > 500
        assert result.duration > 0
        assert sorted(result.speakers) == [5, 9]
    finally:
        result.cleanup()


async def test_solo_isolates_one_speaker(builder, recorded):
    everyone = await builder.build(request_for(recorded, style="mix"))
    alone = await builder.build(request_for(recorded, style="solo", users=[5]))
    try:
        assert alone.speakers == [5]
        # A solo cut must not be the mix in disguise.
        assert alone.path.read_bytes() != everyone.path.read_bytes()
    finally:
        everyone.cleanup()
        alone.cleanup()


async def test_duet_sums_two_speakers(builder, recorded):
    result = await builder.build(request_for(recorded, style="duet", users=[5, 9]))
    try:
        assert result.speakers == [5, 9]
        assert result.path.stat().st_size > 500
    finally:
        result.cleanup()


async def test_duck_changes_the_mix(builder, recorded):
    mixed = await builder.build(request_for(recorded, style="mix"))
    ducked = await builder.build(request_for(recorded, style="duck", users=[5]))
    try:
        assert ducked.path.read_bytes() != mixed.path.read_bytes()
        # Ducking changes balance, not length.
        assert abs(ducked.duration - mixed.duration) < 0.05
    finally:
        mixed.cleanup()
        ducked.cleanup()


async def test_active_style_cuts_the_dead_air(builder, recorded):
    full = await builder.build(request_for(recorded, style="mix"))
    tight = await builder.build(request_for(recorded, style="active"))
    try:
        assert tight.duration < full.duration
    finally:
        full.cleanup()
        tight.cleanup()


async def test_stems_produce_one_file_per_speaker(builder, recorded):
    result = await builder.build(request_for(recorded, style="stems"))
    try:
        # One mix plus one file per speaker.
        assert len(result.all_files()) == 3
        assert all(path.exists() for path in result.all_files())
        assert "all-talkers" in result.filename
    finally:
        result.cleanup()


async def test_stems_respect_an_explicit_speaker_list(builder, recorded):
    result = await builder.build(request_for(recorded, style="stems", users=[5]))
    try:
        assert len(result.all_files()) == 2  # mix plus user 5
        assert result.speakers == [5]
    finally:
        result.cleanup()


async def test_over_long_requests_are_clamped(builder, recorded, config):
    config.audio.max_clip_seconds = 1.0
    result = await builder.build(
        ClipRequest(
            guild_id=1, channel_id=100, session_id=None,
            start=0.0, end=recorded.elapsed(), style="mix", requester_id=1,
        )
    )
    try:
        assert result.duration <= 1.05
    finally:
        result.cleanup()


async def test_a_silent_window_is_refused(builder, recorded):
    """The quiet stretch after a turn must not become an empty clip."""
    # Each block is one second of speech followed by 0.4 s of silence.
    start = FRAMES_PER_BLOCK * 0.02
    window = recorded.read_mix(start, 0.2)
    assert window.size > 0
    assert np.max(np.abs(window)) == 0

    with pytest.raises(LookupError, match="silent"):
        await builder.build(request_for(recorded, start=start, end=start + 0.2))


async def test_an_ignored_speaker_never_reaches_a_ring(sink, config, clock, tmp_path):
    from snippy.voice.recorder import SnippySink

    blind = SnippySink(
        ring_config=config.ring,
        guild_id=1,
        ignored_users=lambda: {7},
        clock=clock,
        wall_clock=clock,
    )
    for _ in range(60):
        feed(blind, clock, speech(9000), 7)
    assert 7 not in blind.rings.known_users()
    assert blind.read_stem(7, 0.0, 1.0) is None
    assert blind.ignored_packets == 60
    # Nothing was ever written, so there is no mixed audio either.
    assert blind.read_mix(0.0, 1.0).size == 0


def test_is_silent_uses_a_sane_threshold():
    assert is_silent(np.zeros((4800, 2), dtype=np.int16))
    assert is_silent(np.full((4800, 2), 3, dtype=np.int16))  # below -60 dBFS
    assert not is_silent(np.full((4800, 2), 4000, dtype=np.int16))
    assert is_silent(np.zeros((0, 2), dtype=np.int16))


def test_orphaned_renders_are_swept_up(builder):
    stale = builder.workdir / "leftover.ogg"
    stale.write_bytes(b"x" * 100)
    old = time.time() - 7200
    os.utime(stale, (old, old))
    assert builder.purge_orphans(older_than=3600.0) == 1
    assert not stale.exists()


def test_fresh_renders_are_left_alone(builder):
    fresh = builder.workdir / "inflight.ogg"
    fresh.write_bytes(b"x" * 100)
    assert builder.purge_orphans(older_than=3600.0) == 0
    assert fresh.exists()
