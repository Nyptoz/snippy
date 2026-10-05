"""Where a clip ends up.

Delivery is configurable per guild because the right answer is different for
every server: some want one channel with everything, some want each call's
clips gathered into a thread, and some want clips to stay private until someone
deliberately posts one. This module resolves that policy and performs the
upload, so the commands only have to say what was clipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import discord

from ..clips import ClipRequest, ClipResult
from ..util.natime import format_duration
from .embeds import DESTINATION_LABELS, clip_embed, error_embed, info_embed
from .views import ClipActions

log = logging.getLogger("snippy.delivery")


class Delivery:
    """Resolves a destination and posts rendered clips."""

    def __init__(self, bot, db, builder) -> None:
        self.bot = bot
        self.db = db
        self.builder = builder
        # session_id -> thread, so a call's clips stay together for the run.
        self._threads: dict[int, discord.Thread] = {}

    # -- destinations --------------------------------------------------------

    async def clip_channel(self, guild: discord.Guild, config) -> discord.TextChannel | None:
        name = config.delivery.clip_channel_name
        for channel in guild.text_channels:
            if getattr(channel, "name", "") == name:
                return channel
        # Create it lazily rather than requiring an admin to do it up front.
        try:
            created = await guild.create_text_channel(
                name, reason="Snippy clip channel", topic="Clips cut by Snippy."
            )
        except discord.Forbidden:
            log.warning("cannot create #%s in %s", name, guild.name)
            return None
        except discord.HTTPException as exc:
            log.warning("could not create #%s in %s: %s", name, guild.name, exc)
            return None
        return created

    async def target_for(
        self,
        guild: discord.Guild,
        config,
        *,
        private: bool,
        voice_channel_id: int | None = None,
    ) -> tuple[discord.abc.Messageable | None, str]:
        """Return where a clip should go, and a description of that choice."""
        destination = config.delivery.destination

        if private:
            if destination == "channel" and not config.delivery.make_public:
                return None, "kept private (the rendered file was discarded)"
            if destination == "dm":
                return None, "sent as a direct message"

        if destination == "dm":
            return None, "will be sent as a direct message"
        if destination == "here":
            channel = self._channel_near(guild, voice_channel_id)
            return channel, "posted in the nearest text channel"
        if destination == "both":
            channel = await self.clip_channel(guild, config)
            return channel, "posted publicly, with a DM copy"
        channel = await self.clip_channel(guild, config)
        return channel, f"posted in #{config.delivery.clip_channel_name}"

    def _channel_near(self, guild: discord.Guild, voice_channel_id: int | None):
        """The text channel most likely to go with this voice channel."""
        if voice_channel_id is not None:
            voice = guild.get_channel(voice_channel_id)
            parent = getattr(voice, "parent", None)
            if isinstance(parent, discord.TextChannel):
                return parent
        for channel in guild.text_channels:
            if not getattr(channel, "category_id", None) and channel.permissions_for(guild.me).send_messages:
                return channel
        return None

    async def thread_for(self, session_id: int | None, parent) -> discord.Thread | None:
        """Fetch or create the thread that collects one call's clips."""
        if session_id is None or parent is None:
            return None
        cached = self._threads.get(session_id)
        if cached is not None:
            return cached
        try:
            thread = await parent.create_thread(
                name=f"Call {session_id}", auto_archive_duration=1440
            )
        except (discord.Forbidden, discord.HTTPException) as exc:
            log.debug("could not create clip thread: %s", exc)
            return None
        self._threads[session_id] = thread
        return thread

    # -- posting -------------------------------------------------------------

    async def post(
        self,
        result: ClipResult,
        request: ClipRequest,
        *,
        guild: discord.Guild,
        config,
        requester: discord.abc.User | None,
        when_text: str,
        transcript: str | None = None,
        destination,
        note: str = "",
    ) -> tuple[discord.Message | None, ClipActions | None, str]:
        """Upload a rendered clip and return the message, its view, and a note."""
        files = [discord.File(str(path), filename=result.filename) for path in result.all_files()]
        lookup = self._name_lookup(guild)
        embed = clip_embed(
            result,
            request=request,
            requester=requester.mention if requester else "someone",
            private=request.private,
            when_text=when_text,
            lookup=lookup,
            transcript=transcript if config.asr.show_transcript else None,
        )
        if note:
            embed.add_field(name="Delivery", value=note, inline=False)

        view = ClipActions(result.clip_id)
        target = destination
        if config.delivery.thread_per_session and target is not None and request.session_id:
            thread = await self.thread_for(request.session_id, target)
            if thread is not None:
                target = thread

        message = None
        if target is not None:
            try:
                message = await target.send(embed=embed, files=files, view=view)
                view.message = message
                await self.db.set_clip_message(result.clip_id, message.id)
            except discord.Forbidden as exc:
                log.warning("no permission to post in %s: %s", target, exc)
            except discord.HTTPException as exc:
                log.warning("could not upload clip: %s", exc)
        return message, view, ""

    async def send_dm(
        self,
        result: ClipResult,
        request: ClipRequest,
        *,
        user: discord.abc.User | None,
        config,
        when_text: str,
        transcript: str | None = None,
    ) -> bool:
        if user is None:
            return False
        try:
            files = [
                discord.File(str(path), filename=result.filename)
                for path in result.all_files()
            ]
            embed = clip_embed(
                result,
                request=request,
                requester="you",
                private=True,
                when_text=when_text,
                transcript=transcript if config.asr.show_transcript else None,
            )
            await user.send(embed=embed, files=files)
            return True
        except discord.Forbidden:
            return False
        except discord.HTTPException as exc:
            log.warning("could not DM clip: %s", exc)
            return False

    # -- helpers -------------------------------------------------------------

    def _name_lookup(self, guild: discord.Guild):
        def lookup(user_id: int) -> str:
            member = guild.get_member(user_id)
            return member.display_name if member else f"user {user_id}"

        return lookup

    def remember_thread(self, session_id: int | None, thread: discord.Thread | None) -> None:
        if session_id is not None and thread is not None:
            self._threads[session_id] = thread

    def forget_session(self, session_id: int) -> None:
        self._threads.pop(session_id, None)

    def cleanup(self) -> None:
        self._threads.clear()


async def report_missing_sink(where_text: str) -> discord.Embed:
    return error_embed(
        f"Snippy is not listening in {where_text} right now.",
        hint="Run `/snippy join`, or ask an admin to set a join policy with `/snippy config join`.",
    )


def window_note(seconds: float) -> str:
    return f"last {format_duration(seconds)}"
