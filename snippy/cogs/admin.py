"""Admin commands.

Everything an administrator might want to change about a server lives here, as
typed slash-command options rather than free text, so the bot can validate a
value before it is stored and can render the current state back as an embed.
"""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from ..clips import STYLES
from ..config import DESTINATIONS, JOIN_POLICIES
from ..store import prune
from ..ui.embeds import error_embed, format_bytes, info_embed, settings_embed
from ..util.natime import format_duration

log = logging.getLogger("snippy.cogs.admin")


class AdminCog(commands.Cog):
    """Settings, privacy, and storage controls."""

    # discord.py has no ``app_commands.group`` decorator. A group is an
    # ``app_commands.Group`` instance; subcommands are attached to it with
    # ``@group.command(...)`` below. Nesting is done by passing ``parent=``,
    # which registers the child with its parent automatically.
    snippy = app_commands.Group(
        name="snippy",
        description="Configure Snippy for this server.",
        default_permissions=discord.Permissions(manage_guild=True),
    )
    config = app_commands.Group(
        name="config",
        description="Change a setting for this server.",
        parent=snippy,
        default_permissions=discord.Permissions(manage_guild=True),
    )

    def __init__(self, bot) -> None:
        self.bot = bot

    def config_for(self, interaction) -> "GuildConfig":
        return self.bot.config_for(interaction.guild.id)

    async def _save(self, interaction, config) -> None:
        problems = config.validate()
        if problems:
            await interaction.response.send_message(
                embed=error_embed(
                    "I did not save that; the result would be invalid.",
                    hint="\n".join(f"• {p}" for p in problems[:4]),
                ),
                ephemeral=True,
            )
            return
        await self.bot.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            embed=info_embed("💾 Saved", "Updated settings for this server."),
            ephemeral=True,
        )

    # -- /snippy -------------------------------------------------------------

    @snippy.command(
        name="settings",
        description="Show the current configuration. Also the /snippy help screen.",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def settings(self, interaction: discord.Interaction) -> None:
        config = self.config_for(interaction)
        await interaction.response.send_message(
            embed=settings_embed(config, self.bot.default_config), ephemeral=True
        )

    # -- /snippy config ------------------------------------------------------

    @config.command(name="audio", description="Clip length, style, and audio processing.")
    @app_commands.describe(
        max_clip_seconds="Hard ceiling on clip length",
        default_window="What /clip grabs with no argument",
        pre_roll="Padding kept before the trigger",
        post_roll="Padding kept after the last word",
        default_style="Default clip style",
        silence_trim="Strip silence from the edges of clips",
        normalize="Even out loudness across speakers",
    )
    @app_commands.choices(
        default_style=[app_commands.Choice(name=s, value=s) for s in STYLES]
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_audio(
        self,
        interaction: discord.Interaction,
        max_clip_seconds: float | None = None,
        default_window: float | None = None,
        pre_roll: float | None = None,
        post_roll: float | None = None,
        default_style: app_commands.Choice[str] | None = None,
        silence_trim: bool | None = None,
        normalize: bool | None = None,
    ) -> None:
        cfg = self.config_for(interaction).audio
        if max_clip_seconds is not None:
            cfg.max_clip_seconds = max_clip_seconds
        if default_window is not None:
            cfg.default_window = default_window
        if pre_roll is not None:
            cfg.pre_roll = pre_roll
        if post_roll is not None:
            cfg.post_roll = post_roll
        if default_style is not None:
            cfg.default_style = default_style.value
        if silence_trim is not None:
            cfg.silence_trim = silence_trim
        if normalize is not None:
            cfg.normalize = normalize
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="delivery", description="Where clips are posted.")
    @app_commands.describe(
        destination="Where clips go",
        clip_channel_name="Name of the clip channel",
        thread_per_session="Group each call's clips into a thread",
        make_public="Allow a private clip to be published later",
    )
    @app_commands.choices(
        destination=[app_commands.Choice(name=d, value=d) for d in DESTINATIONS]
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_delivery(
        self,
        interaction: discord.Interaction,
        destination: app_commands.Choice[str] | None = None,
        clip_channel_name: str | None = None,
        thread_per_session: bool | None = None,
        make_public: bool | None = None,
    ) -> None:
        cfg = self.config_for(interaction).delivery
        if destination is not None:
            cfg.destination = destination.value
        if clip_channel_name is not None:
            cfg.clip_channel_name = clip_channel_name.strip() or "snippy-clips"
        if thread_per_session is not None:
            cfg.thread_per_session = thread_per_session
        if make_public is not None:
            cfg.make_public = make_public
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="join", description="When Snippy joins a voice channel.")
    @app_commands.describe(
        policy="Always, only listed channels, or only on request",
        channels=(
            "A voice channel to listen to. Discord accepts one channel per call, "
            "so run this again to add another."
        ),
        idle_leave_minutes="Leave an empty channel after this many minutes",
    )
    @app_commands.choices(policy=[app_commands.Choice(name=p, value=p) for p in JOIN_POLICIES])
    @app_commands.default_permissions(manage_guild=True)
    async def config_join(
        self,
        interaction: discord.Interaction,
        policy: app_commands.Choice[str] | None = None,
        channels: discord.VoiceChannel | None = None,
        idle_leave_minutes: float | None = None,
    ) -> None:
        cfg = self.config_for(interaction).join
        if policy is not None:
            cfg.policy = policy.value
        if channels is not None:
            # The option is typed as a voice channel, so Discord will not offer a
            # text channel here; the isinstance check stays as a guard against a
            # stage or category sneaking through on a future API change.
            if isinstance(channels, discord.VoiceChannel):
                cfg.channels = [channels.id]
        if idle_leave_minutes is not None:
            cfg.idle_leave_minutes = idle_leave_minutes
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="triggers", description="Spoken and typed trigger phrases.")
    @app_commands.describe(
        spoken="Listen for trigger phrases in voice",
        text="Also react to the trigger phrase typed in chat",
        phrases="Replace the list of trigger phrases",
        fuzzy_threshold="How closely a phrase must match, 0.5 to 1.0",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_triggers(
        self,
        interaction: discord.Interaction,
        spoken: bool | None = None,
        text: bool | None = None,
        phrases: str | None = None,
        fuzzy_threshold: float | None = None,
    ) -> None:
        cfg = self.config_for(interaction).trigger
        if spoken is not None:
            cfg.spoken = spoken
        if text is not None:
            cfg.text = text
        if phrases:
            cfg.phrases = [p.strip() for p in phrases.split(",") if p.strip()]
        if fuzzy_threshold is not None:
            cfg.fuzzy_threshold = fuzzy_threshold
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="archive", description="The rolling recording used by /reclip.")
    @app_commands.describe(
        enabled="Keep a rolling archive at all",
        retention_hours="How long to keep it",
        archive_stems="Also archive each speaker separately (costs one encoder each)",
        stem_retention_hours="How long per-speaker audio is kept",
        segment_seconds="Length of each segment file before rotation",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_archive(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        retention_hours: float | None = None,
        archive_stems: bool | None = None,
        stem_retention_hours: float | None = None,
        segment_seconds: float | None = None,
    ) -> None:
        cfg = self.config_for(interaction).archive
        if enabled is not None:
            cfg.enabled = enabled
        if retention_hours is not None:
            cfg.retention_hours = retention_hours
        if archive_stems is not None:
            cfg.archive_stems = archive_stems
        if stem_retention_hours is not None:
            cfg.stem_retention_hours = stem_retention_hours
        if segment_seconds is not None:
            cfg.segment_seconds = segment_seconds
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="privacy", description="Who is recorded and who may clip.")
    @app_commands.describe(
        allow_solo="Allow clipping one speaker",
        allow_stems="Allow per-speaker clips",
        consent_banner="Post a notice when Snippy starts recording",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_privacy(
        self,
        interaction: discord.Interaction,
        allow_solo: bool | None = None,
        allow_stems: bool | None = None,
        consent_banner: bool | None = None,
    ) -> None:
        cfg = self.config_for(interaction).privacy
        if allow_solo is not None:
            cfg.allow_solo = allow_solo
        if allow_stems is not None:
            cfg.allow_stems = allow_stems
        if consent_banner is not None:
            cfg.consent_banner = consent_banner
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="asr", description="Speech recognition and transcripts.")
    @app_commands.describe(
        enabled="Transcribe speech to detect triggers and caption clips",
        model="Whisper model size, for example tiny.en, base.en, small.en",
        show_transcript="Attach transcripts to clips",
        index_search="Index transcripts so /search can find them",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_asr(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        model: str | None = None,
        show_transcript: bool | None = None,
        index_search: bool | None = None,
    ) -> None:
        cfg = self.config_for(interaction).asr
        if enabled is not None:
            cfg.enabled = enabled
        if model:
            cfg.model = model.strip()
        if show_transcript is not None:
            cfg.show_transcript = show_transcript
        if index_search is not None:
            cfg.index_search = index_search
        await self._save(interaction, self.config_for(interaction))

    @config.command(name="safety", description="Rate limits and retention ceilings.")
    @app_commands.describe(
        clips_per_hour="How many clips one person may make per hour",
        max_retention_hours="Longest archive retention an admin may set",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def config_safety(
        self,
        interaction: discord.Interaction,
        clips_per_hour: int | None = None,
        max_retention_hours: float | None = None,
    ) -> None:
        cfg = self.config_for(interaction).safety
        if clips_per_hour is not None:
            cfg.clips_per_hour = clips_per_hour
        if max_retention_hours is not None:
            cfg.max_retention_hours = max_retention_hours
        await self._save(interaction, self.config_for(interaction))

    # -- /snippy ignore ------------------------------------------------------

    @snippy.command(name="ignore", description="Stop recording someone entirely.")
    @app_commands.describe(member="Who to stop recording")
    @app_commands.default_permissions(manage_guild=True)
    async def ignore(
        self, interaction: discord.Interaction, member: discord.Member, recording: bool = False
    ) -> None:
        config = self.config_for(interaction)
        ignored = set(config.privacy.ignored_users)
        if recording:
            ignored.add(member.id)
        else:
            ignored.discard(member.id)
        config.privacy.ignored_users = sorted(ignored)
        await self.bot.save_config(interaction.guild.id, config)
        verb = "will now" if recording else "is no longer"
        await interaction.response.send_message(
            embed=info_embed(
                "🔇 Updated",
                f"{member.mention} {verb} recorded by Snippy. "
                "Their audio is dropped before it reaches any buffer or the archive.",
            ),
            ephemeral=True,
        )

    # -- /snippy storage -----------------------------------------------------

    @snippy.command(name="storage", description="Show or trim the archive on disk.")
    @app_commands.describe(trim="Delete archive data past the retention window now")
    @app_commands.default_permissions(manage_guild=True)
    async def storage(self, interaction: discord.Interaction, trim: bool = False) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        config = self.config_for(interaction)
        total = await self.bot.db.total_bytes()
        rows = await self.bot.db.all_segments()
        removed = 0
        if trim:
            removed = await prune(
                self.bot.db,
                retention_hours=config.archive.retention_hours,
                quota_mb=config.archive.quota_mb,
            )
            total = await self.bot.db.total_bytes()
        oldest = rows[0]["start_ts"] if rows else None
        embed = info_embed("🗄 Storage", "\n".join(
            filter(None, [
                f"On disk: **{format_bytes(total)}**",
                f"Segments: **{len(rows)}**",
                f"Retention: **{format_duration(config.archive.retention_hours * 3600)}**",
                f"Quota: **{config.archive.quota_mb} MB**",
                f"Deleted: **{removed}**" if trim else None,
            ])
        ))
        await interaction.followup.send(embed=embed, ephemeral=True)

    @snippy.command(name="reset", description="Restore every setting to the defaults.")
    @app_commands.describe(confirm="Must be True. Wipes this server's Snippy settings.")
    @app_commands.default_permissions(manage_guild=True)
    async def reset(self, interaction: discord.Interaction, confirm: bool = False) -> None:
        if not confirm:
            await interaction.response.send_message(
                embed=error_embed(
                    "Nothing was changed.",
                    hint="Run `/snippy reset confirm:True` if you are sure.",
                ),
                ephemeral=True,
            )
            return
        # A reset is the empty diff: everything goes back to the file defaults.
        await self.bot.save_config(interaction.guild.id, self.bot.default_config)
        await interaction.response.send_message(
            embed=info_embed("♻️ Reset", "Every Snippy setting for this server is back to the defaults."),
            ephemeral=True,
        )


async def setup(bot) -> None:
    await bot.add_cog(AdminCog(bot))
