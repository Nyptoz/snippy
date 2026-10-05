"""Ring buffer behaviour: the properties every clip depends on."""

from __future__ import annotations

import numpy as np
import pytest

from snippy.voice.rings import (
    CHANNELS,
    FRAME_SECONDS,
    MIX_BACKFILL_FRAMES,
    SAMPLES_PER_FRAME,
    AudioRing,
    RingSet,
    index_for_time,
    time_for_index,
)


def filled(value: int) -> np.ndarray:
    return np.full((SAMPLES_PER_FRAME, CHANNELS), value, dtype=np.int16)


def test_capacity_matches_requested_seconds():
    ring = AudioRing(2.0)
    assert ring.capacity == int(2.0 / FRAME_SECONDS)


def test_writes_land_at_the_right_index():
    ring = AudioRing(5.0)
    ring.write(10, filled(7))
    ring.write(30, filled(9))
    assert ring.read(10 * FRAME_SECONDS, 1)[0, 0, 0] == 7
    assert ring.read(30 * FRAME_SECONDS, 1)[0, 0, 0] == 9


def test_gaps_are_filled_with_silence():
    """Discord sends nothing while a channel is quiet, and that quiet is real."""
    ring = AudioRing(5.0)
    ring.write(10, filled(7))
    ring.write(30, filled(9))
    block = ring.read(10 * FRAME_SECONDS, 21)
    assert block.shape[0] == 21
    assert np.all(block[1:20] == 0)


def test_stale_and_duplicate_frames_are_rejected():
    ring = AudioRing(5.0)
    assert ring.write(10, filled(1)) is True
    assert ring.write(10, filled(2)) is False
    assert ring.write(3, filled(3)) is False
    assert ring.read(10 * FRAME_SECONDS, 1)[0, 0, 0] == 1


def test_reads_clamp_to_resident_audio():
    ring = AudioRing(5.0)
    for index in range(10):
        ring.write(index, filled(index))
    assert ring.read(0.0, 1000).shape[0] == 10
    assert ring.read(-50.0, 3).shape[0] == 3


def test_wraparound_drops_the_oldest_frames():
    ring = AudioRing(1.0)
    for index in range(ring.capacity * 3):
        ring.write(index, filled(index % 100))
    assert ring.oldest_index == ring.newest_index - ring.capacity
    assert ring.available_seconds() == pytest.approx(1.0, abs=0.01)


def test_mix_sums_every_speaker_on_one_frame():
    """Without this, a mix would only ever contain whoever spoke first."""
    ring = AudioRing(2.0)
    for value in (1000, 2000, 3000):
        ring.accumulate(0, filled(value))
    assert ring.read(0.0, 1)[0, 0, 0] == 6000


def test_mix_accepts_a_late_contributor():
    ring = AudioRing(2.0)
    ring.accumulate(5, filled(100))
    ring.accumulate(4, filled(200))  # arrives after the cursor moved on
    ring.accumulate(5, filled(50))
    assert ring.read(4 * FRAME_SECONDS, 1)[0, 0, 0] == 200
    assert ring.read(5 * FRAME_SECONDS, 1)[0, 0, 0] == 150


def test_mix_rejects_frames_that_are_long_overdue():
    ring = AudioRing(2.0)
    ring.accumulate(MIX_BACKFILL_FRAMES + 5, filled(1))
    assert ring.accumulate(0, filled(9)) is False


def test_mix_saturates_instead_of_wrapping():
    """Summing ten loud speakers must clip, not burst into noise."""
    ring = AudioRing(2.0)
    for _ in range(8):
        ring.accumulate(0, filled(30000))
    assert ring.read(0.0, 1)[0, 0, 0] == 32767


def test_stem_writes_stay_strict():
    ring = AudioRing(2.0)
    assert ring.write(3, filled(1)) is True
    assert ring.write(3, filled(1)) is False


def test_index_and_time_round_trip():
    assert index_for_time(time_for_index(37)) == 37
    # 1.58 / 0.02 is 78.999... in binary floating point.
    assert index_for_time(79 * FRAME_SECONDS) == 79


def test_ring_set_allocates_stems_lazily_and_caps_them(ring_config):
    rings = RingSet(ring_config)
    assert rings.known_users() == []
    for user_id in (1, 2, 3, 4, 5):
        rings.stem_for_write(user_id).write(0, filled(1))
    assert len(rings.known_users()) == ring_config.max_stem_users


def test_ring_set_evicts_the_least_recently_active(ring_config):
    rings = RingSet(ring_config)
    cap = ring_config.max_stem_users
    for user_id in range(cap):
        rings.stem_for_write(user_id).write(0, filled(1))
    rings.stem(0)  # touch user 0 so user 1 is the coldest
    rings.stem_for_write(cap).write(0, filled(1))
    assert 1 not in rings.known_users()
    assert 0 in rings.known_users()


def test_memory_report_grows_with_allocated_stems(ring_config):
    rings = RingSet(ring_config)
    before = rings.memory_bytes()
    for user_id in range(3):
        rings.stem_for_write(user_id).write(0, filled(1))
    assert rings.memory_bytes() > before
