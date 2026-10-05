"""The rolling archive: rotation, indexing hand-off, and speaker runs."""

from __future__ import annotations

import time

import numpy as np
import pytest

from snippy.audio.render import decode_to_pcm
from snippy.voice.archive import ArchiveWriter, SpeechRunTracker

from .conftest import speech

pytestmark = pytest.mark.usefixtures("requires_ffmpeg")


def test_runs_close_when_someone_goes_quiet():
    """A silent user emits no packets, so runs need a stop signal from silence."""
    tracker = SpeechRunTracker()
    start = time.time()
    for offset in range(10):
        tracker.note(5, start + offset * 0.02, True)
    for offset in range(10, 40):
        tracker.note(9, start + offset * 0.02, False)
    tracker.silence(start + 0.8)

    runs = tracker.take(0.0, 1.0)
    assert len(runs) == 1
    assert runs[0]["u"] == 5
    # The run stays open until the channel goes quiet, because a speaker who
    # stops talking simply stops sending packets.
    assert runs[0]["d"] == pytest.approx(0.8, abs=0.05)


def test_a_stuck_run_is_swept_after_a_minute():
    tracker = SpeechRunTracker()
    start = time.time()
    tracker.note(5, start, True)
    tracker.note(5, start + 61.0, True)
    runs = tracker.take(0.0, 120.0)
    assert any(run["d"] <= 61.0 for run in runs)


def test_runs_shorter_than_a_quarter_second_are_ignored():
    tracker = SpeechRunTracker()
    start = time.time()
    tracker.note(5, start, True)
    tracker.silence(start + 0.1)
    assert tracker.take(0.0, 5.0) == []


def test_reset_clears_history():
    tracker = SpeechRunTracker()
    start = time.time()
    tracker.note(5, start, True)
    tracker.silence(start + 1.0)
    assert tracker.take(0.0, 5.0)
    tracker.reset()
    assert tracker.take(0.0, 5.0) == []


def test_writer_reports_when_ffmpeg_is_missing(tmp_path):
    writer = ArchiveWriter(tmp_path, session_id=1, guild_id=1, encoder="/nonexistent/ffmpeg")
    assert writer.available is False
    writer.start()  # must not raise
    writer.feed(speech(9000))
    writer.close()
    assert writer.drain_pending() == []


def test_segments_rotate_and_decode_to_real_audio(tmp_path):
    writer = ArchiveWriter(
        tmp_path, session_id=7, guild_id=1, segment_seconds=0.5, bitrate="32k"
    )
    assert writer.available
    writer.start()
    for _ in range(150):  # 3 seconds, 20 ms per frame
        writer.feed(speech(9000))
    writer.close()

    segments = writer.drain_pending()
    assert len(segments) >= 5  # 0.5 s each
    assert writer.dropped_frames == 0
    for segment in segments:
        assert segment.path and segment.size_bytes > 0
        assert segment.duration == pytest.approx(0.5, abs=0.05)

    pcm = await_decode(segments[0].path)
    assert pcm.shape[0] == pytest.approx(24000, rel=0.02)  # 0.5 s at 48 kHz
    assert np.abs(pcm).max() > 1000


def test_a_session_too_short_to_be_worth_saving_produces_nothing(tmp_path):
    writer = ArchiveWriter(tmp_path, session_id=9, guild_id=1, segment_seconds=60.0)
    writer.start()
    for _ in range(3):
        writer.feed(speech(9000))
    writer.close()
    assert writer.drain_pending() == []


def test_pending_segments_are_handed_over_once(tmp_path):
    writer = ArchiveWriter(tmp_path, session_id=3, guild_id=1, segment_seconds=0.4)
    writer.start()
    for _ in range(80):
        writer.feed(speech(7000))
    writer.close()
    assert writer.drain_pending()
    assert writer.drain_pending() == []


def test_run_provider_is_attached_to_each_segment(tmp_path):
    tracker = SpeechRunTracker()
    base = time.time()
    tracker.note(42, base, True)
    tracker.silence(base + 0.4)

    writer = ArchiveWriter(
        tmp_path,
        session_id=4,
        guild_id=1,
        segment_seconds=0.5,
        runs_provider=lambda start, end: tracker.take(0.0, 10.0),
    )
    writer.start()
    for _ in range(80):
        writer.feed(speech(7000))
    writer.close()

    segments = writer.drain_pending()
    assert segments
    assert all(isinstance(segment.runs, list) for segment in segments)


def await_decode(path):
    import asyncio

    return asyncio.run(decode_to_pcm(path))
