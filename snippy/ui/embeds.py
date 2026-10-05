"""Embeds.

Kept separate from the cogs so the wording and layout can be adjusted without
touching command logic, and so tests can assert on text without a Discord
connection.
"""

from __future__ import annotations

import time

import discord

from ..clips import ClipResult
from ..util.natime import format_duration

BRAND = 0x6CA0FF
MUTED = 0x9AA3B8
WARNING = 0xF2B33D
DANGER = 0xE5646E

STYLE_LABELS = {
    "mix": "Everyone",
    "stems": "Per speaker",
    "solo": "One speaker",
    "duet": "Two speakers",
    "active": "Speech only",
    "duck": "Speaker in front",
}

DESTINATION_LABELS = {
    "channel": "the clip channel",
    "dm": "a direct message",
    "both": "the clip channel and a DM",
    "here": "this channel",
}


def _names(speakers: list[int], lookup) -> str:
    if not speakers:
        return "nobody recorded"
    shown = [lookup(s) for s in speakers[:4]]
    if len(speakers) > 4:
        shown.append(f"+{len(speakers) - 4} more")
    return ", ".join(shown)


def clip_embed(
    result: ClipResult,
    *,
    request,
    requester: str,
    private: bool,
    when_text: str,
    lookup=None,
    transcript: str | None = None,
) -> discord.Embed:
    lookup = lookup or (lambda user_id: f"<@{user_id}>")
    embed = discord.Embed(
        title="🔊 Clip",
        colour=WARNING if private else BRAND,
    )
    embed.description = request.caption or "No caption"
    fields = [
        ("Duration", format_duration(result.duration), True),
        ("Style", STYLE_LABELS.get(result.style, result.style), True),
        ("Visibility", "private" if private else "public", True),
        ("From", when_text, True),
        ("Clip id", f"`{result.clip_id}`", True),
        ("Clipped by", requester, True),
    ]
    if len(result.all_files()) > 1:
        fields.append(("Files", ", ".join(f"`{p.name}`" for p in result.all_files()), False))
    for name, value, inline in fields:
        embed.add_field(name=name, value=value or "-", inline=inline)
    if transcript:
        clipped = transcript[:600] + ("…" if len(transcript) > 600 else "")
        embed.add_field(name="Transcript", value=f"```\n{clipped}\n```", inline=False)
    embed.set_footer(text="Snippy")
    return embed


def error_embed(message: str, *, hint: str | None = None) -> discord.Embed:
    embed = discord.Embed(title="⚠️ Snippy", description=message, colour=DANGER)
    if hint:
        embed.add_field(name="Try", value=hint, inline=False)
    return embed


def info_embed(title: str, message: str, colour: int = BRAND) -> discord.Embed:
    return discord.Embed(title=title, description=message, colour=colour)


def settings_embed(config, defaults=None) -> discord.Embed:
    """A readable dump of the live configuration, with defaults called out."""
    from ..config import flatten

    rows = flatten(config)
    base = flatten(defaults) if defaults is not None else {}
    embed = discord.Embed(
        title="⚙️ Snippy settings",
        colour=BRAND,
    )
    sections: dict[str, list[tuple[str, str, str]]] = {}
    for key, value in rows.items():
        section, _, name = key.partition(".")
        changed = "" if key not in base or base[key] == value else " *(changed)*"
        sections.setdefault(section, []).append((name, value, changed))

    order = [
        ("audio", "Audio"),
        ("delivery", "Delivery"),
        ("join", "Join policy"),
        ("trigger", "Triggers"),
        ("archive", "Archive"),
        ("privacy", "Privacy"),
        ("asr", "Speech recognition"),
        ("ring", "Memory"),
        ("safety", "Safety"),
    ]
    for section, label in order:
        entries = sections.get(section)
        if not entries:
            continue
        body = "\n".join(f"`{name}` {value}{flag}" for name, value, flag in entries)
        embed.add_field(name=label, value=f"```\n{body}```", inline=False)
    return embed


def status_embed(sessions, *, recognizer_line: str, archive_bytes: int, guild_count: int) -> discord.Embed:
    embed = discord.Embed(title="📡 Snippy status", colour=BRAND)
    if not sessions:
        embed.description = "Not listening in any channels right now."
    else:
        for session in sessions:
            name = getattr(session.guild, "name", str(session.guild_id))
            channel = getattr(session.voice_client, "channel", None)
            channel_name = getattr(channel, "name", "unknown")
            embed.add_field(
                name=f"{name} / #{channel_name}",
                value=f"```\n{session.stats_line()}\n```",
                inline=False,
            )
    embed.add_field(
        name="Speech recognition",
        value=recognizer_line,
        inline=True,
    )
    embed.add_field(
        name="Archive on disk",
        value=format_bytes(archive_bytes),
        inline=True,
    )
    embed.add_field(name="Servers", value=str(guild_count), inline=True)
    return embed


def format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def relative(ts: float) -> str:
    """A short, human description of an absolute timestamp."""
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    return f"{format_duration(delta)} ago"
