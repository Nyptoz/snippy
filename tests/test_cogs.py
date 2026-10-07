"""Command-tree tests.

Every other test in this suite exercises logic that does not touch the Discord
command layer, which let two real startup bugs ship: a group decorator that does
not exist in discord.py, and list-typed parameters the Discord API cannot
represent. Both only surface when the cogs are actually imported and their
commands are serialised, so that is what this module does.

No connection is made. The bot is constructed with a fake token and a
temporary data directory, and the command tree is walked in memory.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import asyncio
from types import SimpleNamespace
from typing import Any, get_args, get_origin

import discord
import pytest
from discord import app_commands
from discord.app_commands.transformers import AppCommandOptionType

from snippy.__main__ import build_parser
from snippy.bot import Snippy

COG_PATHS = ("snippy.cogs.clip", "snippy.cogs.admin", "snippy.cogs.voice_cmd")

TOP_LEVEL = {"clip", "reclip", "replay", "when", "search", "session", "snippy"}


def fake_member(user_id: int, name: str) -> discord.Member:
    """A discord.Member with just enough state for id and display_name.

    Building one through the constructor needs a live connection state and a
    full member payload, neither of which exists offline.
    """
    member = object.__new__(discord.Member)
    member._user = SimpleNamespace(
        id=user_id, name=name, discriminator="0", display_name=name
    )
    return member


# -- construction ---------------------------------------------------------


@pytest.fixture
async def bot(tmp_path) -> Snippy:
    """A Snippy that has loaded every cog but never connected."""
    client = Snippy(data_dir=tmp_path / "data", config_path="config/default.toml")
    for name in COG_PATHS:
        await client.load_extension(name)
    return client


def test_every_cog_module_imports():
    """Importing is where the two historical startup bugs surfaced."""
    for path in COG_PATHS:
        __import__(path)


async def test_cogs_are_loaded(bot: Snippy):
    assert set(bot.cogs) == {"ClipCog", "AdminCog", "VoiceCommandCog"}


async def test_top_level_commands(bot: Snippy):
    assert {c.name for c in bot.tree.get_commands()} == TOP_LEVEL


# -- serialisation --------------------------------------------------------


def callback_of(command):
    """The coroutine behind a command. A Group has none."""
    return getattr(command, "callback", None)


def walk(command, path: str = ""):
    """Yield every command in the tree, depth first, as (path, command)."""
    qualified = f"{path} {command.name}".strip()
    yield qualified, command
    if isinstance(command, app_commands.Group):
        for child in command.commands:
            yield from walk(child, qualified)


def all_commands(bot: Snippy):
    for top in bot.tree.get_commands():
        yield from walk(top)


async def test_whole_tree_serialises(bot: Snippy):
    """``to_dict`` is what Discord actually receives, so it must not raise.

    A parameter type the API cannot represent fails here rather than at import,
    which is exactly how the ``list[discord.Member]`` bug stayed hidden.
    """
    tree = bot.tree
    for command in tree.get_commands():
        payload = command.to_dict(tree)
        assert payload["description"], f"{command.name} has no description"
    assert {c.name for c in tree.get_commands()} == TOP_LEVEL


async def test_every_nested_command_has_a_description(bot: Snippy):
    tree = bot.tree

    def check(node, path):
        for option in node.get("options", ()):
            assert option.get("description"), f"{path}/{option['name']} has no description"
            if option.get("type") == 1:  # a subcommand
                check(option, f"{path}/{option['name']}")

    for command in tree.get_commands():
        check(command.to_dict(tree), command.name)


async def test_no_list_typed_parameters(bot: Snippy):
    """Discord has no array option type, so no callback may take a collection."""
    offenders = []
    for qualified, command in all_commands(bot):
        callback = callback_of(command)
        if callback is None:
            continue
        annotations = inspect.get_annotations(callback, eval_str=True)
        for name, param in inspect.signature(callback).parameters.items():
            if name in ("self", "interaction"):
                continue
            origin = get_origin(annotations.get(name, param.annotation))
            if origin in (list, set, tuple, frozenset):
                inner = ", ".join(getattr(a, "__name__", str(a)) for a in get_args(annotations[name]))
                offenders.append(
                    f"{qualified} {name}: {getattr(origin, '__name__', origin)}[{inner}]"
                )
    assert not offenders, "slash commands cannot take collections:\n" + "\n".join(offenders)


async def test_parameter_annotations_are_supported(bot: Snippy):
    """Every annotation must resolve to a real Discord option type.

    The annotation goes through the same ``resolve_annotation`` step discord.py
    applies when it builds a command, so ``X | None`` is normalised the same
    way here as it is at registration.
    """
    from discord.app_commands import transformers
    from discord.utils import resolve_annotation

    valid = set(AppCommandOptionType)
    for qualified, command in all_commands(bot):
        callback = callback_of(command)
        if callback is None:
            continue
        annotations = inspect.get_annotations(callback, eval_str=True)
        scope = callback.__globals__
        for name, param in inspect.signature(callback).parameters.items():
            if name in ("self", "interaction"):
                continue
            resolved = resolve_annotation(
                annotations.get(name, param.annotation), scope, scope, {}
            )
            inner, _, _ = transformers.get_supported_annotation(resolved)
            assert inner.type in valid, f"{qualified} {name} has no option type"


# -- permission placement -------------------------------------------------


async def test_admin_groups_are_manage_guild_only(bot: Snippy):
    """The admin tree must stay behind Manage Server."""
    manage_server = discord.Permissions(manage_guild=True)
    snippy = bot.tree.get_command("snippy")
    assert isinstance(snippy, app_commands.Group)
    assert snippy.default_permissions == manage_server

    config = snippy.get_command("config")
    assert isinstance(config, app_commands.Group)
    assert config.default_permissions == manage_server


async def test_member_commands_are_not_guild_admin_restricted(bot: Snippy):
    """Plain members must be able to clip, so /clip carries no admin gate."""
    assert bot.tree.get_command("clip").default_permissions is None
    assert bot.tree.get_command("replay").default_permissions is None


# -- the specific fixes ---------------------------------------------------


async def test_clip_takes_two_member_slots(bot: Snippy):
    """``solo`` and ``duet`` need two speakers and Discord has no list."""
    clip = bot.tree.get_command("clip")
    assert [p.name for p in clip.parameters] == [
        "window",
        "style",
        "person",
        "person2",
        "caption",
        "private",
    ]
    for name in ("person", "person2"):
        param = next(p for p in clip.parameters if p.name == name)
        assert param.type is AppCommandOptionType.user
        assert not param.required

    reclip = bot.tree.get_command("reclip")
    assert [p.name for p in reclip.parameters][:5] == [
        "when",
        "duration",
        "style",
        "person",
        "person2",
    ]
    assert next(p for p in reclip.parameters if p.name == "when").required


async def test_join_config_takes_one_voice_channel(bot: Snippy):
    join = bot.tree.get_command("snippy").get_command("config").get_command("join")
    channels = next(p for p in join.parameters if p.name == "channels")
    assert channels.type is AppCommandOptionType.channel
    assert [t.name for t in channels.channel_types] == ["voice"]
    assert not channels.required


async def test_session_group_is_a_group(bot: Snippy):
    session = bot.tree.get_command("session")
    assert isinstance(session, app_commands.Group)
    assert {c.name for c in session.commands} == {"join", "leave", "status"}


async def test_user_ids_resolves_speakers(bot: Snippy):
    """The two-slot resolution, including the solo-means-you fallback."""
    cog = bot.get_cog("ClipCog")
    ken = fake_member(111, "Ken")
    alex = fake_member(222, "Alex")

    class FakeInteraction:
        user = ken

    interaction = FakeInteraction()
    assert cog._user_ids(interaction, "solo", ken) == [ken.id]
    assert cog._user_ids(interaction, "duet", ken, alex) == [ken.id, alex.id]
    # A duet given only one speaker is not a duet; refuse rather than guess.
    assert cog._user_ids(interaction, "duet", ken) == []
    # Solo with nobody named means the caller.
    assert cog._user_ids(interaction, "solo", None) == [ken.id]
    assert cog._user_ids(interaction, "mix", ken) == [ken.id]


# -- command registration -------------------------------------------------


class FakeTree:
    """Records what ``sync`` was called with."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def get_commands(self):
        return []

    async def sync(self, guild=None):
        self.calls.append(guild)
        return [object()]


def use_tree(bot: Snippy, tree: FakeTree) -> None:
    """Swap the command tree. discord.py stores it on a name-mangled attribute."""
    setattr(bot, "_BotBase__tree", tree)


async def test_sync_targets_the_configured_guild(bot: Snippy, monkeypatch):
    bot.guild_id = 4242
    guild = SimpleNamespace(id=4242, name="Test Guild")
    monkeypatch.setattr(bot, "get_guild", lambda gid: guild)
    tree = FakeTree()
    use_tree(bot, tree)

    await bot.sync_commands()

    assert tree.calls == [guild], "a guild-scoped sync must not fall back to global"


async def test_missing_guild_does_not_sync_globally(bot: Snippy, monkeypatch):
    bot.guild_id = 4242
    monkeypatch.setattr(bot, "get_guild", lambda gid: None)
    tree = FakeTree()
    use_tree(bot, tree)

    await bot.sync_commands()

    assert tree.calls == [], "an absent guild must not silently register globally"


async def test_no_guild_id_syncs_globally(bot: Snippy):
    bot.guild_id = None
    tree = FakeTree()
    use_tree(bot, tree)

    await bot.sync_commands()

    assert tree.calls == [None]


# -- entrypoint -----------------------------------------------------------


def test_guild_id_flag_defaults_to_env(monkeypatch):
    monkeypatch.setenv("DISCORD_GUILD_ID", "999")
    assert build_parser().parse_args([]).guild_id == "999"


def test_guild_id_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("DISCORD_GUILD_ID", "999")
    assert build_parser().parse_args(["--guild-id", "123"]).guild_id == "123"


def test_main_loads_dotenv_before_reading_the_token(tmp_path, monkeypatch):
    """.env must be read on the live path, not only under ``--check``.

    systemd and run-daemon.sh rely on this: neither exports DISCORD_TOKEN.
    """
    from snippy import __main__ as entry

    (tmp_path / ".env").write_text("DISCORD_TOKEN=token-from-env-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)

    seen: dict[str, Any] = {}

    async def fake_run(args):
        seen["token"] = args.token or os.environ.get("DISCORD_TOKEN")

    def run_coroutine(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(entry, "_run_bot", fake_run)
    monkeypatch.setattr(entry.asyncio, "run", run_coroutine)

    assert entry.main(["--data-dir", str(tmp_path / "data")]) == 0
    assert seen["token"] == "token-from-env-file"


def test_check_uses_the_real_voice_recv_import_path():
    """voice_recv imports as discord.ext.voice_recv, not as a bare module."""
    from snippy import __main__ as entry

    source = inspect.getsource(entry.run_checks)
    assert "discord.ext.voice_recv" in source
    assert '"voice_recv"' not in source
    # And the module really does live there on this install.
    assert importlib.util.find_spec("discord.ext.voice_recv") is not None


# -- startup lifecycle ----------------------------------------------------


async def test_on_ready_reconciles_through_the_session_manager(bot: Snippy):
    """``on_ready`` must call the real SessionManager API.

    It used to call ``self.reconcile_sessions(guild.id)``, a method that has
    never existed. discord.py swallows exceptions raised from ``on_ready`` and
    merely logs them, so the bot logged in, then silently never joined a single
    channel according to the join policy -- invisible without a live gateway.
    """
    calls: list[tuple[str, Any]] = []

    async def fake_sync():
        calls.append(("sync", None))

    async def fake_reconcile(config_for):
        calls.append(("reconcile", config_for))

    bot.sync_commands = fake_sync
    bot.sessions.reconcile = fake_reconcile

    await bot.on_ready()

    assert calls[0][0] == "sync"
    assert [name for name, _ in calls] == ["sync", "reconcile"]
    passed = calls[1][1]
    # reconcile wants the config callback, not a guild id.
    assert inspect.iscoroutinefunction(passed)


def test_on_ready_only_uses_attributes_that_exist(bot: Snippy):
    """Every ``self.X`` referenced by ``on_ready`` must resolve on the instance.

    A cheap structural guard for the whole family of bugs the test above
    caught: discord.py reports a missing method only after a real login.
    """
    source = inspect.getsource(Snippy.on_ready)
    referenced = set(re.findall(r"self\.(\w+)", source))
    missing = sorted(name for name in referenced if not hasattr(bot, name))
    assert missing == [], f"on_ready references attributes that do not exist: {missing}"