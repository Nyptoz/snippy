"""Spoken and typed triggers.

This is the "24/7 friend in the call" half of Snippy. A finished utterance is
transcribed, matched against the intent grammar, and if it reads as a request,
a clip is cut and posted without anybody touching Discord.

Every spoken clip is also posted with the words that triggered it, so a
mis-hearing is visible and correctable rather than mysterious.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import discord
from discord.ext import commands

from ..clips import ClipRequest
from ..intent.matcher import parse_intent
from ..ui.embeds import error_embed, info_embed
from ..util.natime import TimeRef, format_duration, parse_when

log = logging.getLogger("snippy.cogs.voice")

# Ignore an identical request repeated this soon after; people repeat
# themselves when they do not hear the "working" reply.
REPEAT_SUPPRESSION = 6.0


class VoiceCommandCog(commands.Cog):
    """Dispatches spoken intents and typed trigger phrases."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self._last_fired: dict[int, tuple[str, float]] = {}

    async def cog_load(self) -> None:
        # One hook for every session, rather than re-registering per channel.
        for session in self.bot.sessions.sessions.values():
            session.on_intent = self._on_intent
        self.bot.sessions.on_intent = self._on_intent

    # -- spoken --------------------------------------------------------------

    async def _on_intent(self, session, text: str, utterance) -> None:
        config = await self.bot.db.load_config(session.guild_id, self.bot.default_config)
        if not config.trigger.spoken or not config.asr.enabled:
            return

        intent = parse_intent(
            text,
            clip_phrases=config.trigger.phrases,
            replay_phrases=config.trigger.replay_phrases,
            threshold=config.trigger.fuzzy_threshold,
        )
        if not intent.fired:
            return

        if self._recently_fired(session.channel_id, text):
            return
        self._last_fired[session.channel_id] = (text.lower(), time.time())

        speaker = utterance.speaker
        log.info("spoken intent in %s from %s: %r", session.channel_id, speaker, text)

        try:
            if intent.name == "replay":
                await self._replay(session, intent, config)
            else:
                await self._clip(session, intent, config, text, speaker)
        except Exception as exc:
            log.exception("spoken intent failed")
            await self._notify(session, f"I could not do that: {exc}")

    def _recently_fired(self, channel_id: int, text: str) -> bool:
        previous = self._last_fired.get(channel_id)
        if previous is None:
            return False
        last_text, last_time = previous
        return last_text == text.lower() and (time.time() - last_time) < REPEAT_SUPPRESSION

    # -- actions -------------------------------------------------------------

    async def _clip(self, session, intent, config, text: str, speaker: int | None) -> None:
        duration = min(
            intent.duration or config.audio.default_window, config.audio.max_clip_seconds
        )
        end = session.elapsed + config.trigger.spoken_post_roll
        start = max(0.0, end - duration)

        style = intent.style or config.audio.default_style
        users: list[int] = []
        if intent.target:
            resolved = self._resolve_name(session, intent.target)
            if resolved:
                users = [resolved]
        elif style == "solo" and speaker:
            users = [speaker]

        if style in ("solo", "duet") and not users:
            style = "mix"
        if style == "solo" and not config.privacy.allow_solo:
            style = "mix"
        if style in ("stems", "duet") and not config.privacy.allow_stems:
            style = "mix"

        source = "ram"
        segments: list = []
        if intent.when is not None and not _is_recent(intent.when, duration):
            when_ts = intent.when.resolve(time.time(), session.sink.started_wall)
            start = max(0.0, when_ts - duration / 4)
            end = start + duration
            found = await self.bot.find_segments(
                session.guild_id, start, end, users[0] if users else None
            )
            if not found:
                await self._notify(session, f"I do not have {intent.when.describe()} in the archive.")
                return
            source = "archive"
            segments = [
                (_path(row), row["start_ts"], row["start_ts"] + row["duration"]) for row in found
            ]

        private = intent.visibility == "dm"
        request = ClipRequest(
            guild_id=session.guild_id,
            channel_id=session.channel_id,
            session_id=session.db_session_id,
            start=start,
            end=end,
            style=style,
            users=users,
            requester_id=speaker or 0,
            caption=None,
            private=private,
            source=source,
            segments=segments,
            transcript=text,
        )
        if style in ("solo", "duet", "duck") and users:
            names = self.bot.delivery._name_lookup(session.guild)
            request.speaker_names = {u: names(u) for u in users}

        try:
            result = await self.bot.builder.build(request)
        except LookupError as exc:
            await self._notify(session, f"No clip there: {exc}")
            return
        except Exception as exc:
            log.exception("spoken clip failed")
            await self._notify(session, f"Clip failed: {exc}")
            return

        destination, note = await self.bot.delivery.target_for(
            session.guild,
            config,
            private=private,
            voice_channel_id=session.channel_id,
        )
        message, view, _ = await self.bot.delivery.post(
            result,
            request,
            guild=session.guild,
            config=config,
            requester=self.bot.get_user(speaker) if speaker else None,
            when_text=session.voice_client.channel.name if session.voice_client else "voice",
            transcript=text,
            destination=destination,
            note=f"heard: “{text}”",
        )
        if private or config.delivery.destination in ("dm", "both"):
            user = self.bot.get_user(speaker) if speaker else None
            await self.bot.delivery.send_dm(
                result, request, user=user, config=config,
                when_text="just now", transcript=text,
            )
        await self.bot.record_clip(result, request, message, text)
        if config.safety.delete_after_upload:
            result.cleanup()
        log.info("spoken clip delivered to %s (%s)", destination or "dm", note)

    async def _replay(self, session, intent, config) -> None:
        duration = min(intent.duration or 15.0, config.audio.max_clip_seconds)
        when = intent.when or TimeRef("ago", duration)
        if _is_recent(when, duration):
            await self._notify(session, "That is still in my short-term buffer, try again in a moment.")
            return
        when_ts = when.resolve(time.time(), session.sink.started_wall if session.sink else None)
        start = max(0.0, when_ts - min(1.5, duration / 4))
        found = await self.bot.find_segments(session.guild_id, start, start + duration)
        if not found:
            await self._notify(session, f"I do not have {when.describe()} in the archive.")
            return
        request = ClipRequest(
            guild_id=session.guild_id,
            channel_id=session.channel_id,
            session_id=found[0]["session_id"],
            start=start,
            end=start + duration,
            source="archive",
            segments=[(_path(row), row["start_ts"], row["start_ts"] + row["duration"]) for row in found],
        )
        try:
            await session.replay(request)
            await self._notify(session, f"Playing {format_duration(duration)} from {when.describe()}.")
        except Exception as exc:
            log.warning("spoken replay failed: %s", exc)
            await self._notify(session, f"Could not play that: {exc}")

    # -- typed ---------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild or not message.content:
            return
        config = await self.bot.db.load_config(message.guild.id, self.bot.default_config)
        if not config.trigger.text:
            return
        from ..intent.matcher import match_phrase

        matched = match_phrase(
            message.content, config.trigger.phrases, config.trigger.fuzzy_threshold
        )
        if matched is None:
            return
        if self._recently_fired(message.channel.id, message.content):
            return
        self._last_fired[message.channel.id] = (message.content.lower(), time.time())

        session = self._resolve_session(message)
        if session is None or session.sink is None:
            return
        intent = parse_intent(
            message.content,
            clip_phrases=config.trigger.phrases,
            replay_phrases=config.trigger.replay_phrases,
            threshold=config.fuzzy_threshold,
        )
        await self._clip(session, intent, config, message.content, message.author.id)

    def _resolve_session(self, message: discord.Message):
        member = message.guild.get_member(message.author.id)
        if member is not None and member.voice is not None:
            return self.bot.sessions.get(message.guild.id, member.voice.channel.id)
        if isinstance(message.channel, discord.Thread):
            parent = message.channel.parent
            if isinstance(parent, discord.TextChannel) and parent.voice is not None:
                return self.bot.sessions.get(message.guild.id, parent.voice.channel.id)
        sessions = self.bot.sessions.for_guild(message.guild.id)
        return sessions[0] if len(sessions) == 1 else None

    def _resolve_name(self, session, name: str) -> int | None:
        needle = name.lower()
        for member in session.voice_client.channel.members if session.voice_client else []:
            if member.bot:
                continue
            if needle in (member.display_name.lower(), member.name.lower()):
                return member.id
        return None

    async def _notify(self, session, message: str) -> None:
        """Say something in the clip channel, so a spoken request has a receipt."""
        config = await self.bot.db.load_config(session.guild_id, self.bot.default_config)
        target, _note = await self.bot.delivery.target_for(
            session.guild, config, private=False, voice_channel_id=session.channel_id
        )
        if target is None:
            return
        try:
            await target.send(
                embed=info_embed("🎙 Snippy", message), allowed_mentions=discord.AllowedMentions.none()
            )
        except (discord.Forbidden, discord.HTTPException) as exc:
            log.debug("could not post notice: %s", exc)


def _path(row) -> Path:
    return Path(row["path"])


def _is_recent(ref: TimeRef, duration: float) -> bool:
    """True when the reference points inside what the RAM ring still holds."""
    if ref.kind in ("now", "start"):
        return True
    if ref.kind == "ago":
        return ref.value < duration
    return False


async def setup(bot) -> None:
    await bot.add_cog(VoiceCommandCog(bot))
