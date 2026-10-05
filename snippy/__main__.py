"""Command line entrypoint.

``python -m snippy`` runs the bot. ``--check`` validates the environment
without connecting to Discord, which is the fastest way to tell whether a fresh
install actually works.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import load_file_defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snippy", description="A voice clipper that lives in your call."
    )
    parser.add_argument("--token", help="Bot token. Falls back to DISCORD_TOKEN.")
    parser.add_argument("--data-dir", default=os.getenv("SNIPPY_DATA_DIR", "data"))
    parser.add_argument("--config", default=os.getenv("SNIPPY_CONFIG", "config/default.toml"))
    parser.add_argument("--log-level", default=os.getenv("SNIPPY_LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--guild-id",
        default=os.getenv("DISCORD_GUILD_ID"),
        help=(
            "Register slash commands in this one server instead of globally. "
            "Guild commands appear immediately; global ones can take an hour."
        ),
    )
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Enable spoken triggers even if the config file has them off.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the setup and exit without connecting to Discord.",
    )
    parser.add_argument("--version", action="version", version=f"snippy {__version__}")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are chatty at INFO and say nothing a Snippy operator needs.
    for noisy in ("discord.client", "discord.gateway", "discord.http", "websockets", "av"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_checks(args) -> int:
    """Report on everything that can be wrong before the bot ever logs in."""
    problems: list[str] = []
    notes: list[str] = []

    token = args.token or os.getenv("DISCORD_TOKEN")
    if not token or token == "your-bot-token-here":
        problems.append("No bot token. Set DISCORD_TOKEN in .env or pass --token.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        notes.append(f"ffmpeg found at {ffmpeg}")
    else:
        problems.append(
            "ffmpeg is not on PATH. Install it: apt install ffmpeg, or brew install ffmpeg."
        )

    missing: list[str] = []
    # voice_recv ships as discord-ext-voice-recv and imports under
    # discord.ext.voice_recv, not as a top-level module.
    for module in ("discord", "discord.ext.voice_recv", "av", "numpy", "PIL"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        problems.append(
            "Missing packages: " + ", ".join(missing) + ". Run ./install.sh, or pip install -r requirements.txt"
        )
    else:
        notes.append("core dependencies importable")

    try:
        from discord.ext import voice_recv  # noqa: F401

        notes.append("voice receive extension available")
    except ImportError:
        problems.append(
            "discord-ext-voice-recv is missing; Snippy cannot hear anything without it."
        )

    try:
        import faster_whisper  # noqa: F401

        notes.append("speech recognition available")
    except ImportError:
        notes.append(
            "speech recognition not installed; spoken triggers are off. "
            "Run ./install.sh --asr to add it."
        )

    config = load_file_defaults(args.config)
    for problem in config.validate():
        problems.append(f"config: {problem}")
    notes.append(f"config loaded from {args.config}")

    data_dir = Path(args.data_dir)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        notes.append(f"data directory writable at {data_dir.resolve()}")
    except OSError as exc:
        problems.append(f"data directory {data_dir} is not writable: {exc}")

    print(f"Snippy {__version__}\n")
    for note in notes:
        print(f"  ok    {note}")
    for problem in problems:
        print(f"  FAIL  {problem}")
    print()
    if problems:
        print(f"{len(problems)} problem(s) found.")
        return 1
    print("Everything checks out. Start the bot with: python -m snippy")
    return 0


async def _run_bot(args) -> None:
    from .bot import Snippy

    token = args.token or os.getenv("DISCORD_TOKEN")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    guild_id = None
    if args.guild_id:
        try:
            guild_id = int(str(args.guild_id).strip())
        except ValueError:
            print(
                f"--guild-id must be a server id, not {args.guild_id!r}. "
                "Right-click a server in Discord and choose Copy Server ID.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    bot = Snippy(
        data_dir=data_dir,
        config_path=args.config,
        asr_enabled=args.asr,
        guild_id=guild_id,
    )
    try:
        await bot.start(token)
    finally:
        if not bot.is_closed():
            await bot.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    # .env has to be loaded here, before anything reads the environment, so
    # that both `--check` and the live run see DISCORD_TOKEN. The service and
    # daemon launchers do not export it themselves.
    #
    # Both search paths are tried: the working directory, which is the
    # checkout under systemd and both scripts, and the directory holding this
    # package, so it still works when launched from somewhere else. Neither
    # call overrides a variable already present in the environment.
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
        load_dotenv()
    except ImportError:
        pass

    if args.check:
        return run_checks(args)
    if not (args.token or os.getenv("DISCORD_TOKEN")):
        print("No bot token. Copy .env.example to .env and set DISCORD_TOKEN.", file=sys.stderr)
        return 2
    try:
        asyncio.run(_run_bot(args))
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
