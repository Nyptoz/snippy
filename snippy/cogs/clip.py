"""Clip commands.

``/clip`` takes what was just said, ``/reclip`` reaches back into the archive,
``/replay`` plays audio into the voice channel, and ``/when`` draws the map that
the other two navigate by.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from ..clips import ClipRequest
from ..ui.delivery import report_missing_sink
from ..ui.embeds import error_embed, info_embed, relative
from ..util.natime import TimeRef, format_duration, parse_when
from ..audio.timeline import Speaker, render_timeline

log = logging.getLogger("snippy.cogs.clip")

STYLE_CHOICES = [
    app_commands.Choice(name="Everyone (mix)", value="mix"),
    app_commands.Choice(name="Per speaker (stems)", value="stems"),
    app_commands.Choice(name="One speaker (solo)", value="solo"),
    app_commands.Choice(name="Two speakers (duet)", value="duet"),
    app_commands.Choice(name="Speech only (cut the dead air)", value="active"),
    app_commands.Choice(name="Speaker in front (duck others)", value="duck"),
]


def _resolve_voice_channel(interaction, config) -> int | None:
    """Work out which voice channel the request refers to.

    Prefers the caller's own channel, then anyone's in a guild with only one
    busy channel, so `/clip` works without repeating the channel every time.
    """
    member = interaction.user
    if isinstance(member, discord.Member) and member.voice is not None:
        return member.voice.channel.id

    manager = interaction.client.sessions
    sessions = manager.for_guild(interaction.guild.id)
    if len(sessions) == 1:
        return sessions[0].channel_id
    if interaction.channel is not None:
        voice = getattr(interaction.channel, "voice", None)
        if voice is not None:
            return voice.channel.id
        parent = getattr(interaction.channel, "parent", None)
        voice = getattr(parent, "voice", None)
        if voice is not None:
            return voice.channel.id
    return None


class ClipCog(commands.Cog):
    """Everything a member can do."""

    # A group is an ``app_commands.Group`` instance with subcommands attached
    # to it, not a decorator: discord.py 2.x has no ``app_commands.group``.
    session = app_commands.Group(
        name="session",
        description="Control Snippy's presence in a voice channel.",
    )

    def __init__(self, bot) -> None:
        self.bot = bot

    # -- helpers -------------------------------------------------------------

    def config_for(self, interaction) -> "GuildConfig":
        return interaction.client.config_for(interaction.guild.id)

    async def _require_session(self, interaction):
        channel_id = _resolve_voice_channel(interaction, None)
        if channel_id is None:
            return None
        return interaction.client.sessions.get(interaction.guild.id, channel_id)

    async def _rate_limited(self, interaction, config) -> bool:
        allowed = await interaction.client.db.hit_rate_limit(
            "clip",
            f"{interaction.guild.id}:{interaction.user.id}",
            3600.0,
            config.safety.clips_per_hour,
        )
        if not allowed:
            await interaction.response.send_message(
                embed=error_embed(
                    f"You have hit the limit of {config.safety.clips_per_hour} clips per hour.",
                    hint="An admin can raise this with `/snippy config safety`.",
                ),
                ephemeral=True,
            )
        return not allowed

    def _user_ids(
        self,
        interaction,
        request_style: str,
        person: discord.Member | None,
        person2: discord.Member | None = None,
    ) -> list[int]:
        """Resolve the one or two speakers a clip should be isolated to.

        Discord has no array option type, so ``solo`` and ``duet`` take up to
        two separate member options instead of one list.
        """
        ids: list[int] = []
        for candidate in (person, person2):
            if isinstance(candidate, discord.Member):
                ids.append(candidate.id)
            elif isinstance(candidate, discord.User):
                ids.append(candidate.id)
        # A duet with one name is just a solo; do not pass a one-element list
        # where two speakers are expected.
        if len(ids) == 1 and request_style == "duet":
            ids = []
        if not ids and request_style == "solo":
            member = interaction.user
            if isinstance(member, discord.Member):
                ids.append(member.id)
        return ids

    def _check_style_permissions(self, interaction, config, style: str) -> str | None:
        if style == "solo" and not config.privacy.allow_solo:
            return "Clipping a single speaker is disabled here."
        if style in ("stems", "duet") and not config.privacy.allow_stems:
            return "Per-speaker clips are disabled here."
        return None

    # -- /clip ---------------------------------------------------------------

    @app_commands.command(
        name="clip",
        description="Clip what was just said in a voice channel.",
    )
    @app_commands.describe(
        window="How much audio to grab, e.g. 15s, 2m, 1m30s",
        style="How to cut it",
        person="Speaker to isolate, for style One speaker",
        person2="Second speaker, for style Two speakers",
        caption="Optional title for the clip",
        private="Send it to you instead of posting it publicly",
    )
    @app_commands.choices(style=STYLE_CHOICES)
    async def clip(
        self,
        interaction: discord.Interaction,
        window: str = "",
        style: app_commands.Choice[str] | None = None,
        person: discord.Member | None = None,
        person2: discord.Member | None = None,
        caption: str | None = None,
        private: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True)
        config = self.config_for(interaction)
        if await self._rate_limited(interaction, config):
            return

        session = await self._require_session(interaction)
        if session is None or session.sink is None:
            await interaction.followup.send(embed=await report_missing_sink("that channel"), ephemeral=True)
            return

        from ..util.natime import parse_duration

        seconds = parse_duration(window) if window else config.audio.default_window
        if seconds is None:
            await interaction.followup.send(
                embed=error_embed(
                    f"I could not read {window!r} as a length.",
                    hint="Try something like `15s`, `2m`, or `1m30s`.",
                ),
                ephemeral=True,
            )
            return
        seconds = min(seconds, config.audio.max_clip_seconds)

        chosen = (style.value if style else config.audio.default_style) or "mix"
        denied = self._check_style_permissions(interaction, config, chosen)
        if denied:
            await interaction.followup.send(embed=error_embed(denied), ephemeral=True)
            return

        users = self._user_ids(interaction, chosen, person, person2)
        if chosen in ("solo", "duet") and not users:
            await interaction.followup.send(
                embed=error_embed(
                    "Tell me who to isolate.",
                    hint=(
                        "For one speaker: `/clip style:One speaker person:@Ken`. "
                        "For two: `/clip style:Two speakers person:@Ken person2:@Alex`."
                    ),
                ),
                ephemeral=True,
            )
            return

        await self._render_and_send(
            interaction,
            session,
            start=max(0.0, session.elapsed - seconds),
            end=session.elapsed,
            style=chosen,
            users=users,
            caption=caption,
            private=private,
            when_text=f"the last {format_duration(seconds)}",
        )

    # -- /reclip -------------------------------------------------------------

    @app_commands.command(
        name="reclip",
        description="Cut a clip from earlier in the archive.",
    )
    @app_commands.describe(
        when="When to cut from, e.g. 5m, 2 hours ago, yesterday 8pm, 2:30 into the call",
        duration="How long the clip should be",
        style="How to cut it",
        person="Speaker to isolate (needs per-speaker archiving on)",
        person2="Second speaker, for style Two speakers",
        caption="Optional title for the clip",
        private="Send it to you instead of posting it publicly",
    )
    @app_commands.choices(style=STYLE_CHOICES)
    async def reclip(
        self,
        interaction: discord.Interaction,
        when: str,
        duration: str = "20s",
        style: app_commands.Choice[str] | None = None,
        person: discord.Member | None = None,
        person2: discord.Member | None = None,
        caption: str | None = None,
        private: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True)
        config = self.config_for(interaction)
        if not config.archive.enabled:
            await interaction.followup.send(
                embed=error_embed(
                    "The archive is switched off here, so there is nothing to re-clip from.",
                    hint="An admin can enable it with `/snippy config archive enabled:True`.",
                ),
                ephemeral=True,
            )
            return
        if await self._rate_limited(interaction, config):
            return

        from ..util.natime import parse_duration

        seconds = parse_duration(duration) or 20.0
        seconds = min(seconds, config.audio.max_clip_seconds)
        target = parse_when(when)
        if target is None:
            await interaction.followup.send(
                embed=error_embed(
                    f"I could not read {when!r} as a point in time.",
                    hint="Try `5m`, `2 hours ago`, `yesterday 8pm`, or `2:30 into the call`.",
                ),
                ephemeral=True,
            )
            return

        channel_id = _resolve_voice_channel(interaction, config)
        session = (
            self.bot.sessions.get(interaction.guild.id, channel_id) if channel_id else None
        )
        session_start = None
        if session is not None and session.sink is not None:
            session_start = session.sink.started_wall
            if target.kind == "offset":
                target = TimeRef("ago", max(0.0, session.elapsed - target.value), raw=target.raw)

        when_ts = target.resolve(time.time(), session_start)
        # A moment slightly before the target often lands on a better start.
        start_ts = max(0.0, when_ts - min(2.0, seconds / 4))
        end_ts = start_ts + seconds

        chosen = (style.value if style else config.audio.default_style) or "mix"
        denied = self._check_style_permissions(interaction, config, chosen)
        if denied:
            await interaction.followup.send(embed=error_embed(denied), ephemeral=True)
            return
        users = self._user_ids(interaction, chosen, person, person2)

        if chosen in ("solo", "duet") and not config.archive.archive_stems:
            await interaction.followup.send(
                embed=error_embed(
                    "Per-speaker archiving is off, so past clips are mixed only.",
                    hint="An admin can turn it on with `/snippy config archive archive_stems:True`.",
                ),
                ephemeral=True,
            )
            return

        segments = await self.bot.find_segments(
            interaction.guild.id, start_ts, end_ts, users[0] if users else None
        )
        if not segments:
            await interaction.followup.send(
                embed=error_embed(
                    f"Nothing from {target.describe()} is in the archive any more.",
                    hint="The archive keeps the most recent "
                    f"{format_duration(config.archive.retention_hours * 3600)}.",
                ),
                ephemeral=True,
            )
            return

        request = ClipRequest(
            guild_id=interaction.guild.id,
            channel_id=channel_id or 0,
            session_id=segments[0]["session_id"],
            start=start_ts,
            end=end_ts,
            style=chosen,
            users=users,
            requester_id=interaction.user.id,
            caption=caption,
            private=private,
            source="archive",
            segments=[
                (Path(row["path"]), row["start_ts"], row["start_ts"] + row["duration"])
                for row in segments
                if Path(row["path"]).exists()
            ],
        )
        await self._deliver(interaction, request, f"{target.describe()}, {format_duration(seconds)}")

    # -- /replay -------------------------------------------------------------

    @app_commands.command(
        name="replay",
        description="Play an earlier moment back into the voice channel.",
    )
    @app_commands.describe(
        when="When to replay from, e.g. 5m, yesterday 8pm",
        duration="How much to play",
    )
    async def replay(self, interaction: discord.Interaction, when: str, duration: str = "15s") -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        config = self.config_for(interaction)

        from ..util.natime import parse_duration

        seconds = min(parse_duration(duration) or 15.0, config.audio.max_clip_seconds)
        target = parse_when(when)
        if target is None:
            await interaction.followup.send(
                embed=error_embed(
                    f"I could not read {when!r} as a point in time.",
                    hint="Try `5m`, `2 hours ago`, or `yesterday 8pm`.",
                ),
                ephemeral=True,
            )
            return

        session = await self._require_session(interaction)
        if session is None or not session.active:
            await interaction.followup.send(
                embed=error_embed("Snippy is not in a voice channel right now."), ephemeral=True
            )
            return

        session_start = session.sink.started_wall if session.sink else None
        if target.kind == "offset":
            target = TimeRef("ago", max(0.0, session.elapsed - target.value), raw=target.raw)
        when_ts = target.resolve(time.time(), session_start)
        start_ts = max(0.0, when_ts - min(1.5, seconds / 4))
        end_ts = start_ts + seconds

        segments = await self.bot.find_segments(interaction.guild.id, start_ts, end_ts)
        if not segments:
            await interaction.followup.send(
                embed=error_embed(f"Nothing from {target.describe()} is in the archive."), ephemeral=True
            )
            return

        request = ClipRequest(
            guild_id=interaction.guild.id,
            channel_id=session.channel_id,
            session_id=segments[0]["session_id"],
            start=start_ts,
            end=end_ts,
            requester_id=interaction.user.id,
            source="archive",
            segments=[
                (Path(row["path"]), row["start_ts"], row["start_ts"] + row["duration"])
                for row in segments
            ],
        )
        try:
            await session.replay(request)
        except Exception as exc:
            log.warning("replay failed: %s", exc)
            await interaction.followup.send(embed=error_embed(f"Could not play that: {exc}"), ephemeral=True)
            return
        await interaction.followup.send(
            embed=info_embed("🔊 Replaying", f"Playing {format_duration(seconds)} from {target.describe()}."),
            ephemeral=True,
        )

    # -- /when ---------------------------------------------------------------

    @app_commands.command(name="when", description="Show who was talking in a voice channel.")
    @app_commands.describe(window="How far back to look, e.g. 2m, 30s, 10m")
    async def when(self, interaction: discord.Interaction, window: str = "2m") -> None:
        await interaction.response.defer(thinking=True)
        config = self.config_for(interaction)
        session = await self._require_session(interaction)
        if session is None or session.sink is None:
            await interaction.followup.send(embed=await report_missing_sink("that channel"), ephemeral=True)
            return

        from ..util.natime import parse_duration

        seconds = parse_duration(window) or 120.0
        seconds = min(seconds, config.ring.vad_seconds, config.audio.max_clip_seconds)
        elapsed = session.elapsed
        start = max(0.0, elapsed - seconds)
        runs = session.sink.tracker.take(start, elapsed)
        speakers = sorted({int(run["u"]) for run in runs})
        if not speakers:
            await interaction.followup.send(
                embed=info_embed("🗓 Nothing yet", "Nobody has spoken in that window."),
                ephemeral=True,
            )
            return

        labels = {
            user_id: (
                member.display_name
                if (member := interaction.guild.get_member(user_id))
                else f"user {user_id}"
            )
            for user_id in speakers
        }
        buffer = render_timeline(
            runs,
            [Speaker(uid, labels[uid]) for uid in speakers],
            window_start=start,
            window_end=elapsed,
            levels=session.sink.levels_between(start, elapsed),
            title=f"Last {format_duration(elapsed - start)} in #{getattr(session.voice_client.channel, 'name', 'voice')}",
        )
        buffer.name = "voice-activity.png"
        embed = info_embed(
            "🗓 Voice activity",
            "Pick a moment and use `/reclip when:<time>`, or say "
            '"clip that" out loud in the channel.',
        )
        buffer.seek(0)
        await interaction.followup.send(embed=embed, file=discord.File(buffer, "voice-activity.png"))

    # -- /search -------------------------------------------------------------

    @app_commands.command(name="search", description="Search what was said in past clips.")
    @app_commands.describe(query="Words to look for")
    async def search(self, interaction: discord.Interaction, query: str) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        if not self.bot.db.has_fts:
            await interaction.followup.send(
                embed=error_embed(
                    "Transcript search is unavailable on this install.",
                    hint="SQLite was built without FTS5 support.",
                ),
                ephemeral=True,
            )
            return
        config = self.config_for(interaction)
        if not config.asr.index_search:
            await interaction.followup.send(
                embed=error_embed("Transcript indexing is switched off here."), ephemeral=True
            )
            return

        hits = await self.bot.db.search_transcripts(interaction.guild.id, query, limit=8)
        if not hits:
            await interaction.followup.send(
                embed=info_embed("🔍 No matches", f"Nothing in the archive mentions {query!r}."),
                ephemeral=True,
            )
            return
        lines = []
        for hit in hits:
            when = relative(hit["start_ts"]) if hit["start_ts"] else "unknown time"
            excerpt = (hit["text"] or "")[:120]
            jump = (
                f"https://discord.com/channels/{interaction.guild.id}/{hit['message_id']}"
                if hit["message_id"]
                else None
            )
            lines.append(f"**{when}** — {excerpt}" + (f"\n{jump}" if jump else ""))
        await interaction.followup.send(
            embed=info_embed("🔍 Matches", "\n\n".join(lines)[:1800]), ephemeral=True
        )

    # -- /session ------------------------------------------------------------

    @session.command(name="join", description="Ask Snippy to start listening here.")
    async def session_join(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = _resolve_voice_channel(interaction, None)
        if channel_id is None:
            await interaction.followup.send(
                embed=error_embed("I could not tell which voice channel you mean."), ephemeral=True
            )
            return
        config = self.config_for(interaction)
        started = await self.bot.sessions.ensure(interaction.guild.id, channel_id, config)
        if started is None:
            await interaction.followup.send(
                embed=error_embed(
                    "I could not join. Do I have the Connect permission, and is the channel not full?"
                ),
                ephemeral=True,
            )
            return
        await interaction.followup.send(
            embed=info_embed("👂 Listening", f"I am now buffering #{getattr(started.voice_client.channel, 'name', 'the channel')}."),
            ephemeral=True,
        )

    @session.command(name="leave", description="Ask Snippy to stop listening.")
    async def session_leave(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = _resolve_voice_channel(interaction, None)
        await self.bot.sessions.leave(interaction.guild.id, channel_id)
        await interaction.followup.send(embed=info_embed("👋 Left", "I have left that channel."), ephemeral=True)

    @session.command(name="status", description="Show what Snippy is doing right now.")
    async def session_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        embed = await self.bot.status_embed(interaction.guild.id)
        await interaction.followup.send(embed=embed, ephemeral=True)

    # -- shared delivery -----------------------------------------------------

    async def _render_and_send(
        self, interaction, session, *, start, end, style, users, caption, private, when_text
    ) -> None:
        request = session.request_from(
            start=start,
            end=end,
            style=style,
            users=users,
            requester_id=interaction.user.id,
            caption=caption,
            private=private,
        )
        await self._deliver(interaction, request, when_text, session=session)

    async def _deliver(self, interaction, request: ClipRequest, when_text: str, session=None) -> None:
        config = self.config_for(interaction)
        try:
            result = await self.bot.builder.build(request)
        except LookupError as exc:
            await interaction.followup.send(embed=error_embed(str(exc)), ephemeral=True)
            return
        except Exception as exc:
            log.exception("clip render failed")
            await interaction.followup.send(
                embed=error_embed(f"Rendering failed: {exc}"), ephemeral=True
            )
            return

        transcript = request.transcript
        if transcript is None and config.asr.enabled and request.source == "ram":
            transcript = await self.bot.transcribe_clip(request)

        names = self.bot.delivery._name_lookup(interaction.guild)
        request.speaker_names = {uid: names(uid) for uid in result.speakers}

        destination, note = await self.bot.delivery.target_for(
            interaction.guild,
            config,
            private=request.private,
            voice_channel_id=request.channel_id or None,
        )
        message, view, _ = await self.bot.delivery.post(
            result,
            request,
            guild=interaction.guild,
            config=config,
            requester=interaction.user,
            when_text=when_text,
            transcript=transcript,
            destination=destination,
        )

        sent_dm = False
        if config.delivery.destination in ("dm", "both") or request.private:
            sent_dm = await self.bot.delivery.send_dm(
                result,
                request,
                user=interaction.user,
                config=config,
                when_text=when_text,
                transcript=transcript,
            )

        await self.bot.record_clip(result, request, message, transcript)

        if result.path.suffix == ".zip":
            note = f"{note}; stems are in the archive" if note else "stems zipped"

        summary = note
        if request.private and not sent_dm and not message:
            summary = "rendered, but I could not deliver it anywhere"
        elif request.private and sent_dm:
            summary = f"{summary}; DM sent".lstrip("; ")
        await interaction.followup.send(
            embed=info_embed("✅ Clipped", f"{format_duration(result.duration)} — {summary}"),
            ephemeral=True,
        )
        # The rendered copy is disposable: it exists only to be uploaded.
        if config.safety.delete_after_upload:
            result.cleanup()


async def setup(bot) -> None:
    await bot.add_cog(ClipCog(bot))
