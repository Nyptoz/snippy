"""SQLite persistence.

One file, one connection, no ORM, no server. Every public method is async and
funnels through a single worker thread with a lock, which keeps the event loop
free without letting concurrent clip renders corrupt the database.

The interesting table is ``segments``: the archive tier indexes every encoded
chunk of the rolling recording so that "reclip from yesterday at 8pm" is a
lookup rather than a scan of gigabytes of Opus.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import GuildConfig, merge

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS guilds (
    guild_id   INTEGER PRIMARY KEY,
    settings   TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    title       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_guild ON sessions(guild_id, started_at DESC);

CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    guild_id   INTEGER NOT NULL,
    user_id    INTEGER,
    start_ts   REAL NOT NULL,
    duration   REAL NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL DEFAULT 0,
    runs       TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(guild_id, user_id, start_ts);

CREATE TABLE IF NOT EXISTS clips (
    id           TEXT PRIMARY KEY,
    session_id   INTEGER,
    guild_id     INTEGER NOT NULL,
    channel_id   INTEGER,
    requester_id INTEGER,
    start_ts     REAL,
    end_ts       REAL,
    duration     REAL,
    style        TEXT,
    speakers     TEXT,
    source       TEXT,
    path         TEXT,
    filename     TEXT,
    caption      TEXT,
    transcript   TEXT,
    visibility   TEXT NOT NULL DEFAULT 'public',
    message_id   INTEGER,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_created ON clips(guild_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ignores (
    guild_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE IF NOT EXISTS counters (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    key   TEXT NOT NULL,
    ts    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_counters ON counters(scope, key, ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcripts USING fts5(
    text, clip_id UNINDEXED, guild_id UNINDEXED, session_id UNINDEXED,
    start_ts UNINDEXED, end_ts UNINDEXED
);
"""


class Database:
    """Async wrapper around a single SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self.has_fts = False

    # -- lifecycle -----------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.path, check_same_thread=False, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            try:
                self._conn.executescript(FTS_SCHEMA)
                self.has_fts = True
            except sqlite3.OperationalError:
                # FTS5 is compiled into almost every distro's sqlite, but a
                # clip bot should not refuse to start over a missing search
                # index. /search reports the absence instead.
                self.has_fts = False
        return self._conn

    async def init(self) -> None:
        await self._run(lambda _conn: None)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def _run(self, fn, *args):
        def runner():
            with self._lock:
                return fn(self._connection(), *args)

        return await asyncio.to_thread(runner)

    # -- generic helpers -----------------------------------------------------

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await self._run(lambda c, s, p: c.execute(s, p).rowcount, sql, params)

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return await self._run(lambda c, s, p: c.execute(s, p).fetchone(), sql, params)

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return await self._run(lambda c, s, p: c.execute(s, p).fetchall(), sql, params)

    # -- guild settings ------------------------------------------------------

    async def load_config(self, guild_id: int, defaults: GuildConfig) -> GuildConfig:
        """Return a fresh config for this guild, layered over the file defaults.

        Always a new object: callers edit what they get back, and handing out
        the shared defaults instance would let one guild's changes leak into
        every other guild.
        """
        row = await self.fetchone(
            "SELECT settings FROM guilds WHERE guild_id = ?", (guild_id,)
        )
        if row is None:
            return merge(GuildConfig(), defaults.to_dict())
        try:
            stored = json.loads(row["settings"])
        except (TypeError, ValueError):
            return merge(GuildConfig(), defaults.to_dict())
        return merge(defaults, stored)

    async def save_config(self, guild_id: int, config: GuildConfig, changes: dict | None = None) -> None:
        """Store a guild's configuration.

        `changes` is the diff from the shipped defaults. Callers pass it so that
        untouched settings keep following `config/default.toml` after an
        upgrade; passing None stores the full config instead.
        """
        payload = config.to_dict() if changes is None else changes
        await self.execute(
            "INSERT INTO guilds(guild_id, settings, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(guild_id) DO UPDATE SET settings=excluded.settings, "
            "updated_at=excluded.updated_at",
            (guild_id, json.dumps(payload), time.time()),
        )

    async def known_guilds(self) -> list[int]:
        rows = await self.fetchall("SELECT guild_id FROM guilds")
        return [int(r["guild_id"]) for r in rows]

    # -- sessions ------------------------------------------------------------

    async def open_session(
        self, guild_id: int, channel_id: int, started_at: float, title: str | None = None
    ) -> int:
        cursor = await self._run(
            lambda c: c.execute(
                "INSERT INTO sessions(guild_id, channel_id, started_at, title) VALUES(?,?,?,?)",
                (guild_id, channel_id, started_at, title),
            ).lastrowid,
        )
        return int(cursor)

    async def close_session(self, session_id: int, ended_at: float) -> None:
        await self.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (ended_at, session_id)
        )

    async def recent_sessions(self, guild_id: int, limit: int = 10) -> list[sqlite3.Row]:
        return await self.fetchall(
            "SELECT * FROM sessions WHERE guild_id = ? ORDER BY started_at DESC LIMIT ?",
            (guild_id, limit),
        )

    async def session_start(self, session_id: int) -> float | None:
        row = await self.fetchone(
            "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
        )
        return float(row["started_at"]) if row else None

    # -- archive segments ----------------------------------------------------

    async def add_segment(
        self,
        session_id: int,
        guild_id: int,
        user_id: int | None,
        start_ts: float,
        duration: float,
        path: str,
        size_bytes: int,
        runs: list[dict[str, Any]],
    ) -> int:
        return int(
            await self._run(
                lambda c: c.execute(
                    "INSERT INTO segments(session_id, guild_id, user_id, start_ts, duration,"
                    " path, bytes, runs) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        guild_id,
                        user_id,
                        start_ts,
                        duration,
                        str(path),
                        size_bytes,
                        json.dumps(runs),
                    ),
                ).lastrowid,
            )
        )

    async def segments_covering(
        self, guild_id: int, start_ts: float, end_ts: float, user_id: int | None = None
    ) -> list[sqlite3.Row]:
        """Segments that overlap the requested window, in chronological order.

        The comparison is ``start_ts + duration >= wanted_start`` rather than a
        containment test, because a window can begin in the middle of a segment
        that started earlier.
        """
        if user_id is None:
            return await self.fetchall(
                "SELECT * FROM segments WHERE guild_id = ? AND user_id IS NULL"
                " AND start_ts + duration >= ? AND start_ts <= ?"
                " ORDER BY start_ts",
                (guild_id, start_ts, end_ts),
            )
        return await self.fetchall(
            "SELECT * FROM segments WHERE guild_id = ? AND user_id = ?"
            " AND start_ts + duration >= ? AND start_ts <= ?"
            " ORDER BY start_ts",
            (guild_id, user_id, start_ts, end_ts),
        )

    async def expired_segments(self, before_ts: float) -> list[sqlite3.Row]:
        return await self.fetchall(
            "SELECT * FROM segments WHERE start_ts + duration < ? ORDER BY start_ts",
            (before_ts,),
        )

    async def all_segments(self) -> list[sqlite3.Row]:
        return await self.fetchall(
            "SELECT id, start_ts, duration, bytes, path FROM segments ORDER BY start_ts"
        )

    async def delete_segment(self, segment_id: int) -> None:
        await self.execute("DELETE FROM segments WHERE id = ?", (segment_id,))

    async def total_bytes(self, user_id: int | None = None) -> int:
        if user_id is None:
            row = await self.fetchone("SELECT COALESCE(SUM(bytes),0) AS total FROM segments")
        else:
            row = await self.fetchone(
                "SELECT COALESCE(SUM(bytes),0) AS total FROM segments WHERE user_id = ?",
                (user_id,),
            )
        return int(row["total"]) if row else 0

    # -- clips ---------------------------------------------------------------

    async def add_clip(self, record: dict[str, Any]) -> None:
        columns = ", ".join(record)
        placeholders = ", ".join("?" for _ in record)
        await self.execute(
            f"INSERT OR REPLACE INTO clips({columns}) VALUES({placeholders})",
            tuple(record.values()),
        )

    async def get_clip(self, clip_id: str) -> sqlite3.Row | None:
        return await self.fetchone("SELECT * FROM clips WHERE id = ?", (clip_id,))

    async def recent_clips(self, guild_id: int, limit: int = 20) -> list[sqlite3.Row]:
        return await self.fetchall(
            "SELECT * FROM clips WHERE guild_id = ? ORDER BY created_at DESC LIMIT ?",
            (guild_id, limit),
        )

    async def set_clip_message(self, clip_id: str, message_id: int) -> None:
        await self.execute(
            "UPDATE clips SET message_id = ? WHERE id = ?", (message_id, clip_id)
        )

    async def set_clip_visibility(self, clip_id: str, visibility: str) -> None:
        await self.execute(
            "UPDATE clips SET visibility = ? WHERE id = ?", (visibility, clip_id)
        )

    async def delete_clip(self, clip_id: str) -> sqlite3.Row | None:
        row = await self.get_clip(clip_id)
        if row is not None:
            await self.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
            await self.index_transcript("", clip_id, row["guild_id"], None, None, None, remove=True)
        return row

    async def clip_count(self, guild_id: int) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) AS n FROM clips WHERE guild_id = ?", (guild_id,)
        )
        return int(row["n"]) if row else 0

    # -- transcripts ---------------------------------------------------------

    async def index_transcript(
        self,
        text: str,
        clip_id: str,
        guild_id: int,
        session_id: int | None,
        start_ts: float | None,
        end_ts: float | None,
        *,
        remove: bool = False,
    ) -> None:
        if not self.has_fts:
            return
        if remove:
            await self.execute("DELETE FROM transcripts WHERE clip_id = ?", (clip_id,))
            return
        if not text:
            return
        await self.execute(
            "INSERT INTO transcripts(text, clip_id, guild_id, session_id, start_ts, end_ts)"
            " VALUES(?,?,?,?,?,?)",
            (text, clip_id, guild_id, session_id, start_ts, end_ts),
        )

    async def search_transcripts(
        self, guild_id: int, query: str, limit: int = 10
    ) -> list[sqlite3.Row]:
        if not self.has_fts:
            return []
        cleaned = _fts_query(query)
        if not cleaned:
            return []
        try:
            return await self.fetchall(
                "SELECT t.text, t.clip_id, t.start_ts, t.end_ts, c.filename, c.message_id"
                " FROM transcripts t LEFT JOIN clips c ON c.id = t.clip_id"
                " WHERE transcripts MATCH ? AND t.guild_id = ?"
                " ORDER BY bm25(transcripts) LIMIT ?",
                (cleaned, guild_id, limit),
            )
        except sqlite3.OperationalError:
            # A user typed something FTS5 cannot parse. Fall back to a LIKE so
            # /search degrades to substring matching instead of erroring.
            return await self.fetchall(
                "SELECT t.text, t.clip_id, t.start_ts, t.end_ts, c.filename, c.message_id"
                " FROM transcripts t LEFT JOIN clips c ON c.id = t.clip_id"
                " WHERE t.guild_id = ? AND t.text LIKE ? LIMIT ?",
                (guild_id, f"%{query}%", limit),
            )

    # -- privacy -------------------------------------------------------------

    async def ignored_users(self, guild_id: int) -> set[int]:
        rows = await self.fetchall(
            "SELECT user_id FROM ignores WHERE guild_id = ?", (guild_id,)
        )
        return {int(r["user_id"]) for r in rows}

    async def set_ignored(self, guild_id: int, user_id: int, ignored: bool) -> None:
        if ignored:
            await self.execute(
                "INSERT OR IGNORE INTO ignores(guild_id, user_id) VALUES(?,?)",
                (guild_id, user_id),
            )
        else:
            await self.execute(
                "DELETE FROM ignores WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
            )

    # -- rate limiting -------------------------------------------------------

    async def hit_rate_limit(self, scope: str, key: str, window_seconds: float, limit: int) -> bool:
        """Record a hit and report whether it is within the limit.

        Old rows outside the window are dropped on the way through, so the table
        stays small without needing a separate sweeper.
        """
        now = time.time()
        cutoff = now - window_seconds

        def runner(conn: sqlite3.Connection) -> bool:
            with conn:
                conn.execute("DELETE FROM counters WHERE scope=? AND ts < ?", (scope, cutoff))
                count = conn.execute(
                    "SELECT COUNT(*) FROM counters WHERE scope=? AND key=? AND ts >= ?",
                    (scope, key, cutoff),
                ).fetchone()[0]
                if count >= limit:
                    return False
                conn.execute(
                    "INSERT INTO counters(scope, key, ts) VALUES(?,?,?)", (scope, key, now)
                )
                return True

        return await self._run(runner)

    async def leaderboard(self, guild_id: int, limit: int = 10) -> list[tuple[int, int]]:
        """Most-clipped speakers, as ``(user_id, clip_count)`` pairs."""
        rows = await self.fetchall(
            "SELECT speakers, COUNT(*) AS n FROM clips"
            " WHERE guild_id = ? AND speakers IS NOT NULL GROUP BY speakers",
            (guild_id,),
        )
        tally: dict[int, int] = {}
        for row in rows:
            for user_id in json.loads(row["speakers"]):
                tally[int(user_id)] = tally.get(int(user_id), 0) + int(row["n"])
        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]


def _fts_query(text: str) -> str:
    """Turn free user input into a safe FTS5 MATCH expression.

    Everything is quoted so punctuation cannot reach the query parser, which
    would otherwise raise on inputs like ``clip -this`` or an unbalanced quote.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split() if w]
    if not words:
        return ""
    return " AND ".join(f'"{w}"' for w in words)


async def prune(
    db: Database, *, retention_hours: float, quota_mb: int
) -> int:
    """Delete archive segments past the retention window or over quota.

    Returns the number of files removed. Files are unlinked before their index
    rows are dropped, so a crash in between leaves an orphan file rather than an
    index row pointing at nothing.
    """
    cutoff = time.time() - retention_hours * 3600.0
    removed = 0
    for row in await db.expired_segments(cutoff):
        Path(row["path"]).unlink(missing_ok=True)
        await db.delete_segment(int(row["id"]))
        removed += 1

    if quota_mb > 0:
        limit = quota_mb * 1024 * 1024
        rows = await db.all_segments()
        total = sum(int(r["bytes"]) for r in rows)
        for row in rows:
            if total <= limit:
                break
            Path(row["path"]).unlink(missing_ok=True)
            await db.delete_segment(int(row["id"]))
            total -= int(row["bytes"])
            removed += 1
    return removed
