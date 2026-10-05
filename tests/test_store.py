"""Persistence: settings isolation, archive indexing, search, and retention."""

from __future__ import annotations

import json
import time

import pytest

from snippy.config import GuildConfig
from snippy.store import Database, prune


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "test.db")
    await database.init()
    yield database
    database.close()


async def test_config_round_trips(db):
    config = GuildConfig()
    config.audio.max_clip_seconds = 60.0
    config.delivery.destination = "both"
    config.privacy.ignored_users = [7, 8]
    await db.save_config(1, config)

    loaded = await db.load_config(1, GuildConfig())
    assert loaded.audio.max_clip_seconds == 60.0
    assert loaded.delivery.destination == "both"
    assert loaded.privacy.ignored_users == [7, 8]


async def test_guilds_do_not_share_config_objects(db):
    """One guild's edits must never leak into another's settings."""
    defaults = GuildConfig()
    first = await db.load_config(1, defaults)
    first.audio.max_clip_seconds = 42.0
    await db.save_config(1, first)

    second = await db.load_config(2, defaults)
    assert second.audio.max_clip_seconds == defaults.audio.max_clip_seconds
    assert await db.load_config(1, defaults) is not first


async def test_stored_overrides_layer_over_file_defaults(db):
    """Only the diff is persisted, so untouched settings keep following the file.

    Simulates an admin changing one setting, then the operator later correcting
    an unrelated default in config/default.toml.
    """
    from snippy.config import diff_from

    shipped = GuildConfig()
    stored = GuildConfig()
    stored.audio.max_clip_seconds = 30.0
    await db.save_config(5, stored, diff_from(shipped, stored))

    corrected_shipped = GuildConfig()
    corrected_shipped.audio.bitrate = "96k"
    loaded = await db.load_config(5, corrected_shipped)
    assert loaded.audio.max_clip_seconds == 30.0
    assert loaded.audio.bitrate == "96k"


async def test_a_full_save_freezes_every_value(db):
    defaults = GuildConfig()
    stored = GuildConfig()
    stored.audio.max_clip_seconds = 30.0
    await db.save_config(6, stored)
    defaults.audio.bitrate = "96k"
    assert (await db.load_config(6, defaults)).audio.bitrate == "48k"


async def test_segment_lookup_finds_overlap_not_containment(db):
    """A window starting mid-segment must still find that segment."""
    session = await db.open_session(1, 100, time.time() - 3600)
    base = time.time() - 3600
    await db.add_segment(session, 1, None, base, 300, "a.ogg", 10, [])
    await db.add_segment(session, 1, None, base + 300, 300, "b.ogg", 10, [])

    found = await db.segments_covering(1, base + 250, base + 350)
    assert [row["path"] for row in found] == ["a.ogg", "b.ogg"]

    assert await db.segments_covering(1, base - 5000, base - 4000) == []


async def test_per_speaker_segments_are_separate_from_the_mix(db):
    session = await db.open_session(1, 100, time.time())
    base = time.time() - 60
    await db.add_segment(session, 1, None, base, 300, "mix.ogg", 10, [])
    await db.add_segment(session, 1, 42, base, 300, "u42.ogg", 10, [])

    mixed = await db.segments_covering(1, base - 1, base + 1)
    solo = await db.segments_covering(1, base - 1, base + 1, user_id=42)
    assert [r["path"] for r in mixed] == ["mix.ogg"]
    assert [r["path"] for r in solo] == ["u42.ogg"]


async def test_ignore_list_round_trips(db):
    await db.set_ignored(1, 42, True)
    assert await db.ignored_users(1) == {42}
    await db.set_ignored(1, 42, False)
    assert await db.ignored_users(1) == set()


async def test_rate_limiter_allows_then_blocks(db):
    assert await db.hit_rate_limit("clip", "u1", 3600, 2) is True
    assert await db.hit_rate_limit("clip", "u1", 3600, 2) is True
    assert await db.hit_rate_limit("clip", "u1", 3600, 2) is False
    # A different subject has its own budget.
    assert await db.hit_rate_limit("clip", "u2", 3600, 2) is True


async def test_rate_limiter_window_expires(db):
    assert await db.hit_rate_limit("clip", "u1", 0.05, 1) is True
    assert await db.hit_rate_limit("clip", "u1", 0.05, 1) is False
    time.sleep(0.1)
    assert await db.hit_rate_limit("clip", "u1", 0.05, 1) is True


@pytest.fixture
def searchable_db(db):
    if not db.has_fts:
        pytest.skip("sqlite was built without FTS5")
    return db


async def test_transcript_search_finds_and_ranks(searchable_db):
    await searchable_db.index_transcript(
        "the whole thing fell off the table", "c1", 1, 10, 1.0, 2.0
    )
    await searchable_db.index_transcript(
        "unrelated chatter about weather", "c2", 1, 10, 3.0, 4.0
    )
    hits = await searchable_db.search_transcripts(1, "table")
    assert [h["clip_id"] for h in hits] == ["c1"]


async def test_search_never_leaks_across_guilds(searchable_db):
    await searchable_db.index_transcript("secret words", "c1", 1, 10, 1.0, 2.0)
    assert await searchable_db.search_transcripts(2, "secret") == []


async def test_search_survives_punctuation(searchable_db):
    """FTS5 would raise on unbalanced quotes; the query must stay safe."""
    for query in ['clip -this "x', "a OR (b", "***", "'; DROP TABLE clips; --"]:
        assert await searchable_db.search_transcripts(1, query) == []


async def test_deleting_a_clip_removes_its_transcript(searchable_db):
    await searchable_db.add_clip(
        dict(
            id="gone", session_id=1, guild_id=1, channel_id=1, requester_id=1,
            start_ts=1.0, end_ts=2.0, duration=1.0, style="mix", speakers="[1]",
            source="ram", path="x.ogg", filename="x.ogg", caption=None,
            transcript="vanishing words", visibility="public", message_id=None,
            created_at=time.time(),
        )
    )
    await searchable_db.index_transcript("vanishing words", "gone", 1, 1, 1.0, 2.0)
    assert await searchable_db.search_transcripts(1, "vanishing")
    await searchable_db.delete_clip("gone")
    assert await searchable_db.search_transcripts(1, "vanishing") == []


async def test_leaderboard_counts_each_speaker(db):
    for clip_id, speakers in (("a", "[1, 2]"), ("b", "[2]"), ("c", "[1]")):
        await db.add_clip(
            dict(
                id=clip_id, session_id=1, guild_id=1, channel_id=1, requester_id=1,
                start_ts=1.0, end_ts=2.0, duration=1.0, style="mix", speakers=speakers,
                source="ram", path="x.ogg", filename="x.ogg", caption=None,
                transcript=None, visibility="public", message_id=None,
                created_at=time.time(),
            )
        )
    assert dict(await db.leaderboard(1)) == {1: 2, 2: 2}


async def test_prune_deletes_expired_files_and_keeps_recent(db, tmp_path):
    session = await db.open_session(1, 100, time.time() - 100 * 3600)
    old_path = tmp_path / "old.ogg"
    old_path.write_bytes(b"x" * 100)
    new_path = tmp_path / "new.ogg"
    new_path.write_bytes(b"y" * 100)

    await db.add_segment(session, 1, None, time.time() - 100 * 3600, 300, str(old_path), 100, [])
    await db.add_segment(session, 1, None, time.time() - 60, 300, str(new_path), 100, [])

    removed = await prune(db, retention_hours=1, quota_mb=0)
    assert removed == 1
    assert not old_path.exists()
    assert new_path.exists()


async def test_prune_trims_to_the_quota(db, tmp_path):
    session = await db.open_session(1, 100, time.time())
    paths = []
    for index in range(4):
        path = tmp_path / f"seg{index}.ogg"
        path.write_bytes(b"z" * 1000)
        paths.append(path)
        await db.add_segment(session, 1, None, time.time() - 400 + index * 100, 100, str(path), 1000, [])

    await prune(db, retention_hours=99, quota_mb=0.001)  # ~1 KB
    remaining = await db.all_segments()
    assert len(remaining) < 4
    assert await db.total_bytes() <= 1000


async def test_sessions_track_start_and_end(db):
    session = await db.open_session(1, 100, time.time() - 60, "call")
    await db.close_session(session, time.time())
    rows = await db.recent_sessions(1)
    assert rows[0]["ended_at"] is not None
