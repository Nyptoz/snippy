"""Per-channel session state.

One :class:`VoiceSession` exists for every voice channel Snippy is listening
to, and each one owns a voice connection, a capture sink, an archive writer, and
the background tasks that keep them honest. :class:`SessionManager` owns the
set of them and decides when to join, when to leave, and which one a command
should act on.

The recurring hazard in this file is the voice receive stall. The underlying
extension occasionally stops reading from the gateway socket with no error and
no disconnect, which looks identical to a room that has simply gone quiet. The
watchdog distinguishes the two by looking at packet arrival, not connection
state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.render import ffmpeg_path
from ..clips import ClipBuilder, ClipRequest, ClipResult
from ..config import RingConfig
from .archive import ArchiveWriter
from .rings import FRAME_SECONDS, SAMPLES_PER_FRAME
from .recorder import HAVE_AV, SnippySink

log = logging.getLogger("snippy.session")

# How often the archive pump moves finished mix audio into the encoder.
ARCHIVE_TICK = 0.25
# Watchdog cadence, and the silence after which a still-but-connected session
# is considered stuck rather than quiet.
WATCHDOG_TICK = 10.0
STALL_SECONDS = 20.0
# How long the ASR worker waits between utterance checks.
UTTERANCE_TICK = 0.5


def scale_ring_config(ring: RingConfig, active_sessions: int) -> RingConfig:
    """Shrink the per-session ring so many channels cannot exhaust memory.

    The budget is shared, so a server with six busy channels gets six smaller
    windows rather than six full-size ones and an out-of-memory kill.
    """
    active_sessions = max(1, active_sessions)
    if ring.memory_budget_mb <= 0:
        return ring
    per_session = (ring.memory_budget_mb * 1024 * 1024) / active_sessions
    # Mix and stems are int16 stereo at 48 kHz: 192 000 bytes per second each.
    mix_bytes = per_session * 0.55
    stem_bytes = per_session * 0.30
    mix_seconds = mix_bytes / 192_000.0
    stem_seconds = stem_bytes / (192_000.0 * max(1, ring.max_stem_users))
    return RingConfig(
        mix_seconds=max(10.0, min(ring.mix_seconds, round(mix_seconds, 1))),
        stem_seconds=max(5.0, min(ring.stem_seconds, round(stem_seconds, 1))),
        vad_seconds=ring.vad_seconds,
        max_stem_users=ring.max_stem_users,
        memory_budget_mb=ring.memory_budget_mb,
    )


@dataclass
class SessionStats:
    packets: int = 0
    dropped: int = 0
    gaps: int = 0
    decode_failures: int = 0
    archived: float = 0.0
    reconnects: int = 0


class VoiceSession:
    """Snippy's presence in one voice channel."""

    def __init__(
        self,
        *,
        bot,
        guild_id: int,
        channel_id: int,
        config,
        db,
        builder: ClipBuilder,
        archive_dir: Path,
        recognizer=None,
        active_sessions: int = 1,
    ) -> None:
        self.bot = bot
        self.guild = bot.get_guild(guild_id)
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.config = config
        self.db = db
        self.builder = builder
        self.archive_dir = Path(archive_dir)
        self.recognizer = recognizer
        self.on_intent = None  # set by the manager
        self.manager = None
        self.active_sessions = max(1, active_sessions)

        self.voice_client = None
        self.sink: SnippySink | None = None
        self.db_session_id: int | None = None
        self.archive: ArchiveWriter | None = None
        self.stem_archives: dict[int, ArchiveWriter] = {}
        self.stats = SessionStats()

        self.connected_at: float | None = None
        self.last_packet_at: float = 0.0
        self.last_human_at: float = 0.0
        self._archive_cursor = 0
        self._tasks: list[asyncio.Task] = []
        self._closing = False
        self._reconnecting = False

    # -- state ---------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.voice_client is not None and not self._closing

    @property
    def elapsed(self) -> float:
        return self.sink.elapsed() if self.sink else 0.0

    def should_listen(self) -> bool:
        if self.sink is None:
            return False
        members = [
            m for m in self.voice_client.channel.members
            if not m.bot
        ]
        return bool(members)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        channel = self.guild.get_channel(self.channel_id)
        if channel is None:
            raise RuntimeError("voice channel no longer exists")

        ring_config = scale_ring_config(self.config.ring, self.active_sessions)
        self.db_session_id = await self.db.open_session(
            self.guild_id, self.channel_id, time.time(), self.guild.name
        )

        try:
            from discord.ext import voice_recv
        except Exception as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "discord-ext-voice-recv is not installed; Snippy cannot receive voice"
            ) from exc

        if not HAVE_AV:
            raise RuntimeError("PyAV is missing, so Opus cannot be decoded")

        voice_client = await channel.connect(
            cls=voice_recv.VoiceRecvClient, self_deaf=True
        )
        self.voice_client = voice_client

        self.sink = SnippySink(
            ring_config=ring_config,
            guild_id=self.guild_id,
            ignored_users=lambda: self._ignored,
            min_utterance_ms=self.config.trigger.min_utterance_ms,
            on_packet=self._on_packet,
        )
        voice_client.listen(self.sink)
        self.connected_at = time.time()
        self.last_packet_at = time.time()
        self.last_human_at = time.time()

        if self.config.archive.enabled and ffmpeg_path():
            self.archive = ArchiveWriter(
                self.archive_dir,
                session_id=self.db_session_id,
                guild_id=self.guild_id,
                bitrate=self.config.archive.bitrate,
                segment_seconds=self.config.archive.segment_seconds,
                runs_provider=self._runs_for,
            )
            self.archive.start()

        self._spawn(self._archive_pump())
        self._spawn(self._watchdog())
        self._spawn(self._utterance_worker())
        if self.config.join.idle_leave_minutes > 0:
            self._spawn(self._idle_watch())
        log.info(
            "listening in %s / #%s (session %s, mix %.0fs, stems %.0fs x%d)",
            self.guild.name,
            getattr(channel, "name", self.channel_id),
            self.db_session_id,
            ring_config.mix_seconds,
            ring_config.stem_seconds,
            ring_config.max_stem_users,
        )

    async def stop(self, *, reason: str = "left") -> None:
        if self._closing:
            return
        self._closing = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass
        self._tasks.clear()

        if self.archive is not None:
            self.archive.close()
            await self._index_segments(self.archive.drain_pending())
        for writer in self.stem_archives.values():
            writer.close()
        self.stem_archives.clear()

        if self.sink is not None:
            self.sink.cleanup()
        if self.voice_client is not None:
            try:
                await self.voice_client.disconnect(force=True)
            except Exception as exc:  # pragma: no cover
                log.debug("disconnect from %s raised %s", self.channel_id, exc)
        if self.db_session_id is not None:
            await self.db.close_session(self.db_session_id, time.time())
        log.info("stopped session in %s (%s)", self.channel_id, reason)

    def _spawn(self, coro) -> None:
        self._tasks.append(asyncio.create_task(coro))

    @property
    def _ignored(self) -> set[int]:
        return set(self.config.privacy.ignored_users)

    def _on_packet(self, user_id: int, index: int) -> None:
        self.last_packet_at = time.time()
        self.stats.packets += 1
        if self.config.archive.archive_stems and self.archive is not None:
            sink = self.sink
            if sink is None:
                return
            writer = self.stem_archives.get(user_id)
            if writer is None:
                writer = ArchiveWriter(
                    self.archive_dir,
                    session_id=self.db_session_id or 0,
                    guild_id=self.guild_id,
                    user_id=user_id,
                    bitrate=self.config.archive.bitrate,
                    segment_seconds=self.config.archive.segment_seconds,
                )
                writer.start()
                self.stem_archives[user_id] = writer
            block = sink.read_stem(user_id, index * FRAME_SECONDS, FRAME_SECONDS)
            if block is not None and block.size:
                writer.feed(block)

    def _runs_for(self, start_ts: float, end_ts: float) -> list[dict]:
        """Adapt the session-relative speech runs the archive asks for."""
        if self.sink is None:
            return []
        window_start = start_ts - self.sink.started_wall
        window_end = end_ts - self.sink.started_wall
        return self.sink.tracker.take(window_start, window_end)

    # -- background tasks ----------------------------------------------------

    async def _archive_pump(self) -> None:
        """Move finished mix audio into the archive encoder.

        Reading the mix ring rather than feeding the archive per packet keeps
        the capture path free of any encoder latency, and guarantees the archive
        holds the same thing the mix does regardless of who spoke first.
        """
        while not self._closing:
            try:
                await asyncio.sleep(ARCHIVE_TICK)
                if self.archive is None or self.sink is None:
                    continue
                newest = self.sink.rings.mix.newest_index
                pending = newest - self._archive_cursor
                if pending <= 0:
                    continue
                # Stay a little behind the cursor so every frame is complete.
                batch = self.sink.rings.mix.read(
                    self._archive_cursor * FRAME_SECONDS, pending
                )
                if batch.size:
                    flat = self.sink.rings.mix.to_pcm(batch)
                    total = flat.shape[0] // SAMPLES_PER_FRAME
                    for frame_index in range(total):
                        start = frame_index * SAMPLES_PER_FRAME
                        self.archive.feed(flat[start : start + SAMPLES_PER_FRAME])
                    self._archive_cursor += batch.shape[0]
                await self._index_segments(self.archive.drain_pending())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                log.warning("archive pump error in %s: %s", self.channel_id, exc)

    async def _index_segments(self, segments) -> None:
        for segment in segments:
            await self.db.add_segment(
                segment.session_id,
                segment.guild_id,
                segment.user_id,
                segment.start_ts,
                segment.duration,
                segment.path,
                segment.size_bytes,
                segment.runs,
            )

    async def _watchdog(self) -> None:
        """Reconnect a session that stopped receiving without disconnecting."""
        while not self._closing:
            try:
                await asyncio.sleep(WATCHDOG_TICK)
                if self.voice_client is None or self._reconnecting:
                    continue
                if time.time() - self.last_packet_at < STALL_SECONDS:
                    continue
                if not self.should_listen():
                    # A genuinely empty room. The idle watcher will handle it.
                    continue
                if self.voice_client.is_connected():
                    log.warning(
                        "no audio received for %ds in %s; reconnecting",
                        STALL_SECONDS,
                        self.channel_id,
                    )
                    await self._reconnect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                log.warning("watchdog error in %s: %s", self.channel_id, exc)

    async def _reconnect(self) -> None:
        self._reconnecting = True
        try:
            if self.voice_client is not None:
                try:
                    await self.voice_client.disconnect(force=True)
                except Exception:
                    pass
            channel = self.guild.get_channel(self.channel_id)
            if channel is None:
                return
            from discord.ext import voice_recv

            self.voice_client = await channel.connect(
                cls=voice_recv.VoiceRecvClient, self_deaf=True
            )
            if self.sink is not None:
                # Keep the decoders; clear the audio so a clip cannot span a gap
                # that has nothing to do with what anybody said.
                self.sink.reset()
                self._archive_cursor = 0
                self.voice_client.listen(self.sink)
            self.last_packet_at = time.time()
            self.stats.reconnects += 1
        finally:
            self._reconnecting = False

    async def _idle_watch(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(30.0)
                if self.should_listen():
                    self.last_human_at = time.time()
                    continue
                idle = (time.time() - self.last_human_at) / 60.0
                if idle >= self.config.join.idle_leave_minutes:
                    log.info("leaving empty channel %s after %.0fm", self.channel_id, idle)
                    await self.manager_leave()
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                log.warning("idle watch error in %s: %s", self.channel_id, exc)

    async def manager_leave(self) -> None:
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager.leave(self.guild_id, self.channel_id)
        else:  # pragma: no cover
            await self.stop()

    async def _utterance_worker(self) -> None:
        """Transcribe finished utterances and dispatch spoken intents."""
        while not self._closing:
            try:
                await asyncio.sleep(UTTERANCE_TICK)
                if self.sink is None or self.recognizer is None:
                    continue
                if not self.config.trigger.spoken or not self.config.asr.enabled:
                    continue
                if not self.recognizer.available:
                    continue
                for utterance in self.sink.drain_utterances():
                    await self._handle_utterance(utterance)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                log.warning("utterance worker error in %s: %s", self.channel_id, exc)

    async def _handle_utterance(self, utterance) -> None:
        duration = utterance.end - utterance.start
        max_seconds = self.config.trigger.max_utterance_seconds
        if duration <= 0 or duration > max_seconds:
            return
        audio = self.sink.read_activity(utterance.start, duration)
        if audio.size == 0:
            return
        text = await self.recognizer.transcribe(audio)
        if not text:
            return
        utterance.text = text
        if self.on_intent is not None:
            await self.on_intent(self, text, utterance)

    # -- clip operations -----------------------------------------------------

    def request_from(self, **kwargs) -> ClipRequest:
        kwargs.setdefault("guild_id", self.guild_id)
        kwargs.setdefault("channel_id", self.channel_id)
        kwargs.setdefault("session_id", self.db_session_id)
        kwargs.setdefault("source", "ram")
        return ClipRequest(**kwargs)

    async def clip(self, request: ClipRequest) -> ClipResult:
        return await self.builder.build(request)

    async def replay(self, request: ClipRequest) -> None:
        """Play a rendered clip back into the voice channel."""
        result = await self.builder.build(request)
        try:
            import discord

            if self.voice_client is None:
                raise RuntimeError("Snippy is not in the voice channel")
            source = discord.FFmpegOpusAudio(str(result.path))
            self.voice_client.play(source)
        finally:
            if self.config.safety.delete_after_upload:
                result.cleanup()

    def stats_line(self) -> str:
        sink = self.sink
        parts = [f"up {int(time.time() - self.connected_at) if self.connected_at else 0}s"]
        if sink is not None:
            parts.append(f"{sink.packets} packets")
            parts.append(f"{sink.ignored_packets} ignored")
            if sink.rings.mix.gap_frames:
                parts.append(f"{sink.rings.mix.gap_frames} gap frames")
            if sink.decode_failures:
                parts.append(f"{sink.decode_failures} decode errors")
        if self.archive is not None:
            parts.append(f"{self.archive.encoded_seconds:.0f}s archived")
            if self.archive.dropped_frames:
                parts.append(f"{self.archive.dropped_frames} archive drops")
        if self.stats.reconnects:
            parts.append(f"{self.stats.reconnects} reconnects")
        return ", ".join(parts)


class SessionManager:
    """Owns every session and the policy for when Snippy should be present."""

    def __init__(self, *, bot, db, builder: ClipBuilder, data_dir: Path, recognizer=None) -> None:
        self.bot = bot
        self.db = db
        self.builder = builder
        self.archive_dir = Path(data_dir) / "archive"
        self.recognizer = recognizer
        self.sessions: dict[tuple[int, int], VoiceSession] = {}
        self._joining: set[tuple[int, int]] = set()
        builder.sink_provider = self.sink_for
        builder.sessions = self.sessions

    # -- lookup --------------------------------------------------------------

    def get(self, guild_id: int, channel_id: int) -> VoiceSession | None:
        return self.sessions.get((guild_id, channel_id))

    def sink_for(self, guild_id: int, channel_id: int) -> SnippySink | None:
        session = self.get(guild_id, channel_id)
        return session.sink if session else None

    def for_guild(self, guild_id: int) -> list[VoiceSession]:
        return [s for (g, _), s in self.sessions.items() if g == guild_id]

    def session_for_voice_channel(self, voice_channel_id: int) -> VoiceSession | None:
        for session in self.sessions.values():
            if session.channel_id == voice_channel_id:
                return session
        return None

    # -- lifecycle -----------------------------------------------------------

    async def ensure(self, guild_id: int, channel_id: int, config) -> VoiceSession | None:
        """Join a channel if not already there. Safe to call repeatedly."""
        key = (guild_id, channel_id)
        if key in self.sessions or key in self._joining:
            return self.sessions.get(key)

        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return None
        channel = guild.get_channel(channel_id)
        if channel is None or not hasattr(channel, "connect"):
            return None
        if not self.bot.permissions_for(guild).connect:
            log.info("no Connect permission in %s; skipping", guild.name)
            return None

        self._joining.add(key)
        session = VoiceSession(
            bot=self.bot,
            guild_id=guild_id,
            channel_id=channel_id,
            config=config,
            db=self.db,
            builder=self.builder,
            archive_dir=self.archive_dir,
            recognizer=self.recognizer,
            # Count this session too, so the memory plan is right from the
            # moment the first channel joins rather than the second.
            active_sessions=len(self.sessions) + 1,
        )
        session.manager = self
        try:
            await session.start()
        except Exception as exc:
            log.warning("could not join %s in %s: %s", channel_id, guild.name, exc)
            self._joining.discard(key)
            return None
        self.sessions[key] = session
        self._joining.discard(key)
        return session

    async def leave(self, guild_id: int, channel_id: int) -> None:
        key = (guild_id, channel_id)
        session = self.sessions.pop(key, None)
        if session is not None:
            await session.stop()

    async def leave_all(self) -> None:
        for key in list(self.sessions):
            await self.leave(*key)

    async def on_voice_state_update(self, member) -> None:
        """Apply the join policy when someone starts talking in a channel."""
        channel_id = getattr(member, "voice_channel", None)
        if channel_id is None or getattr(member, "bot", False):
            return
        guild_id = member.guild.id
        config = await self.db.load_config(guild_id, self.bot.default_config)
        policy = config.join.policy
        if policy == "ondemand":
            return
        if policy == "whitelist" and channel_id not in config.join.channels:
            return
        if (guild_id, channel_id) in self.sessions:
            return
        await self.ensure(guild_id, channel_id, config)

    async def reconcile(self, config_for) -> None:
        """Join or leave channels so the bot matches the current join policy."""
        for guild in self.bot.guilds:
            config = await config_for(guild.id)
            wanted: set[int] = set()
            for channel in guild.voice_channels:
                humans = [m for m in channel.members if not m.bot]
                if not humans:
                    continue
                if config.join.policy == "always":
                    wanted.add(channel.id)
                elif config.join.policy == "whitelist" and channel.id in config.join.channels:
                    wanted.add(channel.id)
            for channel in guild.voice_channels:
                key = (guild.id, channel.id)
                if key in self.sessions and channel.id not in wanted:
                    await self.leave(guild.id, channel.id)
            for channel_id in wanted:
                await self.ensure(guild.id, channel_id, config)
