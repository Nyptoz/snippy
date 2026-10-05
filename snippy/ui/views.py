"""Interactive controls on posted clips.

Every clip comes back with a row of buttons so the awkward bits can be fixed
after the fact instead of before: trim the edges, hear it again, flip it from
private to public, or get rid of it. The clip file is gone from disk by the
time somebody clicks anything, so each action re-renders from the stored
request rather than re-uploading what was already sent.
"""

from __future__ import annotations

import logging

import discord

log = logging.getLogger("snippy.ui")

# Trim presets, as (label, new_start_offset, new_duration) relative to the
# original clip. "Trim head" and "Trim tail" are the two people actually want.
TRIM_OPTIONS = [
    ("✂️ Trim 2s off the start", 2.0, None),
    ("✂️ Trim 5s off the start", 5.0, None),
    ("✂️ Trim 2s off the end", 0.0, -2.0),
    ("✂️ Trim 5s off the end", 0.0, -5.0),
    ("🎯 Keep the first half", 0.0, 0.5),
    ("🎯 Keep the second half", 0.5, 0.5),
    ("↩️ Restore the full clip", 0.0, 1.0),
]


class ClipActions(discord.ui.View):
    """Buttons attached to every clip message."""

    def __init__(self, clip_id: str, *, timeout: float = 900.0) -> None:
        super().__init__(timeout=timeout)
        self.clip_id = clip_id
        self.message: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True

    def disable_all(self, note: str = "expired") -> None:
        for item in self.children:
            if hasattr(item, "disabled"):
                item.disabled = True
        self.timeout = 0.01
        log.debug("clip view %s %s", self.clip_id, note)

    @discord.ui.button(label="Replay in voice", emoji="🔊", style=discord.ButtonStyle.secondary)
    async def replay(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        bot = interaction.client
        handler = getattr(bot, "handle_clip_action", None)
        if handler is None:
            await interaction.followup.send("Replaying is not available right now.", ephemeral=True)
            return
        ok, message = await handler(self.clip_id, "replay", interaction)
        await interaction.followup.send(message, ephemeral=True)

    @discord.ui.button(label="Make public", emoji="📢", style=discord.ButtonStyle.secondary)
    async def publish(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        bot = interaction.client
        handler = getattr(bot, "handle_clip_action", None)
        if handler is None:
            await interaction.followup.send("Publishing is not available right now.", ephemeral=True)
            return
        ok, message = await handler(self.clip_id, "publish", interaction)
        await interaction.followup.send(message, ephemeral=True)

    @discord.ui.button(label="Delete", emoji="🗑️", style=discord.ButtonStyle.danger)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        bot = interaction.client
        handler = getattr(bot, "handle_clip_action", None)
        if handler is None:
            await interaction.followup.send("Deleting is not available right now.", ephemeral=True)
            return
        ok, message = await handler(self.clip_id, "delete", interaction)
        if ok and self.message is not None:
            try:
                await self.message.delete()
            except discord.HTTPException as exc:  # pragma: no cover
                log.debug("could not delete clip message: %s", exc)
        await interaction.followup.send(message, ephemeral=True)

    @discord.ui.select(
        placeholder="Trim the clip…",
        options=[
            discord.SelectOption(label=label, value=f"{start}|{duration}")
            for label, start, duration in TRIM_OPTIONS
        ],
    )
    async def trim(self, interaction: discord.Interaction, select: discord.ui.Select) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        bot = interaction.client
        handler = getattr(bot, "handle_clip_action", None)
        if handler is None:
            await interaction.followup.send("Trimming is not available right now.", ephemeral=True)
            return
        try:
            start_ratio, duration_ratio = (float(p) for p in select.values[0].split("|"))
        except (ValueError, IndexError):
            await interaction.followup.send("I did not understand that trim.", ephemeral=True)
            return
        ok, message = await handler(self.clip_id, "trim", interaction, (start_ratio, duration_ratio))
        await interaction.followup.send(message, ephemeral=True)
