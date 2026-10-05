"""The bot: wiring, lifecycle, and the shared operations commands need.

Everything that more than one command needs lives here, so the cogs stay thin:
config lookup with caching, archive searches, clip bookkeeping, and the
background maintenance that keeps the archive inside its retention window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import discord
from discord.ext import commands

from .clips import ClipBuilder, ClipRequest
from .config import GuildConfig, load_file_defaults
from .intent.asr import Recognizer
from .store import Database, prune
from .ui.delivery import Delivery
from .ui.embeds import error_embed, status_embed
from .util.natime import format_duration
from .voice.session import SessionManager

log = logging.getLogger("snippy.bot")

# How often expired archive segments are swept up.
MAINTENANCE_INTERVAL = 900.0


class Snippy(commands.Bot):
    """The Discord client and the owner of every long-lived object."""

    def __init__(
        self,
        *,
        token: str | None = None,
        data_dir: str | Path = "data",
        config_path: str | Path = "config/default.toml",
        asr_enabled: bool = False,
        guild_id: int | None = None,
    ) -> None:
        intents = discord.Intents.default()
        # Needed to hear the trigger phrase when it is typed in a channel, and
        # to resolve member names for solo and stem clips.
        intents.message_content = True
        # Snippy is slash-command only; the prefix exists because discord.py
        # still requires one.
        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "archive").mkdir(exist_ok=True)
        (self.data_dir / "clips").mkdir(exist_ok=True)

        # When set, slash commands are registered in this one server, which
        # takes effect immediately instead of the up to an hour a global
        # registration can take.
        self.guild_id = guild_id

        self.default_config: GuildConfig = load_file_defaults(config_path)
        self.config_cache: dict[int, tuple[float, GuildConfig]] = {}
        self.db = Database(self.data_dir / "snippy.db")
        self.builder = ClipBuilder(self.data_dir, self.default_config, self.db)
        self.delivery = Delivery(self, self.db, self.builder)
        self.sessions = SessionManager(
            bot=self, db=self.db, builder=self.builder, data_dir=self.data_dir
        )
        self.recognizer = Recognizer(self._asr_config(asr_enabled))
        self.sessions.recognizer = self.recognizer
        self._maintenance: asyncio.Task | None = None
        self.started_at = time.time()

    def _asr_config(self, override: bool):
        from dataclasses import replace

        asr = self.default_config.asr
        if override and not asr.enabled:
            asr = replace(asr, enabled=True)
        return asr

    # -- configuration -------------------------------------------------------

    async def config_for(self, guild_id: int) -> GuildConfig:
        """Per-guild settings, cached briefly so a busy channel is not re-read."""
        cached = self.config_cache.get(guild_id)
        if cached is not None and time.time() - cached[0] < 30.0:
            return cached[1]
        config = await self.db.load_config(guild_id, self.default_config)
        self.config_cache[guild_id] = (time.time(), config)
        return config

    def invalidate_config(self, guild_id: int) -> None:
        self.config_cache.pop(guild_id, None)

    # -- lifecycle -----------------------------------------------------------

    async def setup_hook(self) -> None:
        await self.db.init()
        for name, path in (
            ("clip", "snippy.cogs.clip"),
            ("admin", "snippy.cogs.admin"),
            ("voice", "snippy.cogs.voice_cmd"),
        ):
            try:
                await self.load_extension(path)
            except Exception as exc:
                log.error("could not load %s cog: %s", name, exc)
        self._maintenance = asyncio.create_task(self._maintenance_loop())

    async def on_ready(self) -> None:
        log.info("logged in as %s", self.user)
        await self.sync_commands()
        for guild in self.guilds:
            await self.reconcile_sessions(guild.id)
        self._announce_asr()

    async def sync_commands(self) -> None:
        """Register slash commands, in one guild when one is configured.

        Guild-scoped commands appear right away. A global sync can take up to
        an hour to propagate, so a single-guild deployment never falls back to
        one by accident: if the configured guild is missing, that is reported
        loudly instead of silently registering everywhere.
        """
        try:
            if self.guild_id is None:
                synced = await self.tree.sync()
                log.info("synced %d slash commands globally", len(synced))
                return

            guild = self.get_guild(self.guild_id)
            if guild is None:
                log.error(
                    "guild %s is not available; was the bot invited to it? "
                    "Commands were not registered. Fix DISCORD_GUILD_ID or unset it "
                    "to register globally.",
                    self.guild_id,
                )
                return
            synced = await self.tree.sync(guild=guild)
            log.info("synced %d slash commands to %s", len(synced), guild.name)
        except Exception as exc:
            log.warning("command sync failed: %s", exc)

    def _announce_asr(self) -> None:
        line = self.recognizer.status_line
        if self.recognizer.available:
            log.info("spoken triggers: %s", line)
        else:
            log.info("spoken triggers off (%s). Slash commands still work.", line)

    async def on_voice_state_update(self, member) -> None:
        if member.bot:
            return
        try:
            await self.sessions.on_voice_state_update(member)
        except Exception as exc:
            log.warning("voice state handling failed: %s", exc)

    async def close(self) -> None:
        if self._maintenance is not None:
            self._maintenance.cancel()
        await self.sessions.leave_all()
        self.recognizer.close()
        self.db.close()
        await super().close()

    # -- maintenance ---------------------------------------------------------

    async def _maintenance_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(MAINTENANCE_INTERVAL)
                await self.run_maintenance()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                log.warning("maintenance error: %s", exc)

    async def run_maintenance(self) -> None:
        """Sweep expired archive segments and orphaned render files."""
        removed = 0
        for guild in self.guilds:
            config = await self.config_for(guild.id)
            if not config.archive.enabled:
                continue
            removed += await prune(
                self.db,
                retention_hours=config.archive.retention_hours,
                quota_mb=config.archive.quota_mb,
            )
        orphans = self.builder.purge_orphans(older_than=3600.0)
        if removed or orphans:
            log.info("maintenance removed %d segments and %d orphaned files", removed, orphans)

    # -- shared operations ---------------------------------------------------

    async def find_segments(self, guild_id: int, start: float, end: float, user_id: int | None = None):
        return await self.db.segments_covering(guild_id, start, end, user_id)

    async def transcribe_clip(self, request: ClipRequest) -> str | None:
        """Transcribe a freshly cut clip so it can be captioned and indexed."""
        if not self.recognizer.available or request.source != "ram":
            return None
        session = self.sessions.get(request.guild_id, request.channel_id)
        if session is None or session.sink is None:
            return None
        audio = session.sink.read_activity(request.start, request.duration)
        if audio.size == 0:
            return None
        return await self.recognizer.transcribe(audio) or None

    async def save_config(self, guild_id: int, config: GuildConfig) -> None:
        """Persist only what differs from the shipped defaults."""
        from .config import diff_from

        await self.db.save_config(guild_id, config, diff_from(self.default_config, config))
        self.invalidate_config(guild_id)

    async def record_clip(self, result, request: ClipRequest, message, transcript: str | None) -> None:
        import json

        await self.db.add_clip(
            {
                "id": result.clip_id,
                "session_id": request.session_id,
                "guild_id": request.guild_id,
                "channel_id": request.channel_id,
                "requester_id": request.requester_id,
                "start_ts": request.start,
                "end_ts": request.end,
                "duration": result.duration,
                "style": result.style,
                "speakers": json.dumps(result.speakers),
                "source": request.source,
                "path": str(result.path),
                "filename": result.filename,
                "caption": request.caption,
                "transcript": transcript,
                "visibility": "private" if request.private else "public",
                "message_id": getattr(message, "id", None),
                "created_at": time.time(),
            }
        )
        config = await self.config_for(request.guild_id)
        if transcript and config.asr.index_search:
            await self.db.index_transcript(
                transcript,
                result.clip_id,
                request.guild_id,
                request.session_id,
                request.start,
                request.end,
            )

    async def status_embed(self, guild_id: int | None = None) -> discord.Embed:
        sessions = (
            self.sessions.for_guild(guild_id) if guild_id else list(self.sessions.sessions.values())
        )
        return status_embed(
            sessions,
            recognizer_line=self.recognizer.status_line,
            archive_bytes=await self.db.total_bytes(),
            guild_count=len(self.guilds),
        )

    # -- clip button actions -------------------------------------------------

    async def handle_clip_action(
        self, clip_id: str, action: str, interaction, trim=None
    ) -> tuple[bool, str]:
        row = await self.db.get_clip(clip_id)
        if row is None:
            return False, "That clip is no longer in the index, so there is nothing to act on."

        try:
            if action == "delete":
                await self.db.delete_clip(clip_id)
                return True, "Deleted."

            config = await self.config_for(int(row["guild_id"]))
            request = ClipRequest(
                guild_id=int(row["guild_id"]),
                channel_id=int(row["channel_id"] or 0),
                session_id=row["session_id"],
                start=float(row["start_ts"] or 0.0),
                end=float(row["end_ts"] or 0.0),
                style=row["style"] or "mix",
                users=[],
                requester_id=row["requester_id"],
                caption=row["caption"],
                private=row["visibility"] == "private",
                source=row["source"] or "ram",
            )
            if action == "trim" and trim is not None:
                start_ratio, duration_ratio = trim
                original = request.end - request.start
                request.start += original * start_ratio
                request.end = request.start + original * duration_ratio
            if request.source == "archive":
                rows = await self.find_segments(request.guild_id, request.start, request.end)
                request.segments = [
                    (Path(r["path"]), r["start_ts"], r["start_ts"] + r["duration"])
                    for r in rows
                    if Path(r["path"]).exists()
                ]

            result = await self.builder.build(request)
        except LookupError as exc:
            return False, str(exc)
        except Exception as exc:
            log.exception("clip action %s failed", action)
            return False, f"That did not work: {exc}"

        try:
            if action == "replay":
                session = self.sessions.get(request.guild_id, request.channel_id)
                if session is None or not session.active:
                    return False, "Snippy is not in that voice channel right now."
                if request.source == "archive":
                    import discord as _discord

                    if session.voice_client is None:
                        return False, "Snippy is not connected to voice."
                    session.voice_client.play(_discord.FFmpegOpusAudio(str(result.path)))
                else:
                    session.voice_client.play(discord.FFmpegOpusAudio(str(result.path)))
                return True, f"Playing {format_duration(result.duration)} in voice."

            if action == "publish":
                guild = self.get_guild(request.guild_id)
                if guild is None:
                    return False, "I am not in that server any more."
                destination, _note = await self.delivery.target_for(
                    guild, config, private=False, voice_channel_id=request.channel_id or None
                )
                if destination is None:
                    return False, "I could not find a channel to post into."
                request.private = False
                message, _view, _ = await self.delivery.post(
                    result,
                    request,
                    guild=guild,
                    config=config,
                    requester=interaction.user,
                    when_text="an earlier clip",
                    transcript=row["transcript"],
                    destination=destination,
                )
                await self.db.set_clip_visibility(clip_id, "public")
                if message is not None:
                    await self.db.set_clip_message(clip_id, message.id)
                return True, "Posted it publicly." if message else "I could not post it."

            if action == "trim":
                guild = self.get_guild(request.guild_id)
                destination = interaction.channel
                message, _view, _ = await self.delivery.post(
                    result,
                    request,
                    guild=guild,
                    config=config,
                    requester=interaction.user,
                    when_text="a trimmed version",
                    transcript=row["transcript"],
                    destination=destination,
                )
                return True, f"Trimmed to {format_duration(result.duration)} and posted it here."
            return False, "I did not understand that action."
        finally:
            if config.safety.delete_after_upload:
                result.cleanup()

    # -- text fallback -------------------------------------------------------

    async def on_command_error(self, context, exception) -> None:
        if isinstance(exception, commands.CommandNotFound):
            return
        log.error("command error: %s", exception, exc_info=context.invoked_with_parent_command is None)
        if context.channel is not None:
            try:
                await context.send(
                    embed=error_embed("Something went wrong running that command.", hint=str(exception)[:200])
                )
            except discord.HTTPException:
                pass


async def run(token: str, **kwargs) -> None:
    bot = Snippy(token=token, **kwargs)
    try:
        await bot.start(token)
    finally:
        if not bot.is_closed():
            await bot.close()
