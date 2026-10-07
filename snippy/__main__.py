"""Command line entrypoint.

``python -m snippy`` runs the bot. ``--check`` validates the environment
without connecting to Discord, which is the fastest way to tell whether a fresh
install actually works.

On a real terminal the check report draws a braille loading animation while
each probe runs and colors its verdict; piped or redirected output keeps the
plain ``ok``/``FAIL`` layout so scripts can grep it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import threading
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


# -- terminal styling ------------------------------------------------------
#
# Everything here degrades to nothing: when stdout is a pipe, a file, or a
# console that refuses VT processing, no escape codes are ever emitted.

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
MAGENTA = "\x1b[35m"


def _enable_vt(stream) -> bool:
    """Ask Windows to interpret ANSI escapes on this stream.

    Legacy conhost renders nothing (or mojibake) until virtual terminal
    processing is switched on for the handle; Windows Terminal already has it.
    """
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _fancy(stream) -> bool:
    """True when colors and animation will actually render on this stream."""
    if os.getenv("NO_COLOR") is not None or os.getenv("TERM") == "dumb":
        return False
    if not stream.isatty():
        return False
    if os.name == "nt":
        return _enable_vt(stream)
    return True


class _Spinner:
    """A braille loading animation drawn on a background thread.

    The animation runs while a check blocks (importing discord and numpy takes
    a second on a cold cache), then the verdict line replaces it. All writes
    are serialized so the check results never interleave with a frame.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    TICK = 0.09

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._active = False
        self._hidden = False
        self._label = ""
        self._frame = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- drawing ------------------------------------------------------------

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            # A closed or oddly-encoded stream must never take the check down.
            pass

    def _run(self) -> None:
        while not self._done.wait(self.TICK):
            with self._lock:
                if not self._active:
                    continue
                self._frame = (self._frame + 1) % len(self.FRAMES)
                self._paint(self.FRAMES[self._frame])

    def _paint(self, char: str) -> None:
        self._write(f"\r\x1b[2K  {char} {DIM}{self._label}{RESET}")

    def begin(self, label: str) -> None:
        """Show the animation for the step that is about to run."""
        with self._lock:
            self._label = label
            self._frame = 0
            self._active = True
            if not self._hidden:
                self._write("\x1b[?25l")
                self._hidden = True
            self._paint(self.FRAMES[0])

    def result(self, ok: bool, message: str) -> None:
        """Replace the animation line with the finished verdict."""
        with self._lock:
            self._active = False
            if ok:
                self._write(f"\r\x1b[2K  {GREEN}✓ ok{RESET}    {message}\n")
            else:
                self._write(f"\r\x1b[2K  {RED}✗ FAIL{RESET}  {message}\n")

    def close(self) -> None:
        """Stop animating, clear any half-drawn line, and show the cursor."""
        with self._lock:
            self._active = False
            if self._hidden:
                self._write("\r\x1b[2K")
                self._write("\x1b[?25h")
                self._hidden = False
        self._done.set()
        self._thread.join(timeout=2.0)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if _fancy(sys.stderr):
        logging.getLogger().handlers[0].setFormatter(_ColorFormatter())
    # These libraries are chatty at INFO and say nothing a Snippy operator needs.
    for noisy in ("discord.client", "discord.gateway", "discord.http", "websockets", "av"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _ColorFormatter(logging.Formatter):
    """The default layout with the level colored and loud messages tinted.

    A red traceback reads much faster than a black one, so WARNING and above
    color the whole message; INFO keeps its text neutral. The plain format is
    inherited from ``configure_logging`` so redirected logs look unchanged.
    """

    _LEVEL = {
        "DEBUG": DIM,
        "INFO": CYAN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": "\x1b[1;31m",
    }
    _BODY = {"WARNING": YELLOW, "ERROR": RED, "CRITICAL": "\x1b[1;31m"}

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        level_color = self._LEVEL.get(record.levelname)
        if level_color is None:
            return text
        stamp, _, rest = text.partition(" ")
        level, _, message = rest.partition(" ")
        body_color = self._BODY.get(record.levelname, "")
        return f"{DIM}{stamp}{RESET} {level_color}{level}{RESET} {body_color}{message}{RESET}"


def run_checks(args) -> int:
    """Report on everything that can be wrong before the bot ever logs in."""
    problems: list[str] = []
    notes: list[str] = []
    live = _fancy(sys.stdout)
    spin = _Spinner(sys.stdout) if live else None

    def ok(message: str) -> None:
        notes.append(message)
        if spin is not None:
            spin.result(True, message)

    def fail(message: str) -> None:
        problems.append(message)
        if spin is not None:
            spin.result(False, message)

    if live:
        print(f"{BOLD}{MAGENTA}Snippy {__version__}{RESET}\n")

    try:
        if spin is not None:
            spin.begin("looking for a bot token")
        token = args.token or os.getenv("DISCORD_TOKEN")
        if not token or token == "your-bot-token-here":
            fail("No bot token. Set DISCORD_TOKEN in .env or pass --token.")

        if spin is not None:
            spin.begin("looking for ffmpeg")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            ok(f"ffmpeg found at {ffmpeg}")
        else:
            fail(
                "ffmpeg is not on PATH. Install it: apt install ffmpeg, or brew install ffmpeg."
            )

        if spin is not None:
            spin.begin("importing the core packages")
        missing: list[str] = []
        # voice_recv ships as discord-ext-voice-recv and imports under
        # discord.ext.voice_recv, not as a top-level module.
        for module in ("discord", "discord.ext.voice_recv", "av", "numpy", "PIL"):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if missing:
            fail(
                "Missing packages: "
                + ", ".join(missing)
                + ". Run ./install.sh, or pip install -r requirements.txt"
            )
        else:
            ok("core dependencies importable")

        if spin is not None:
            spin.begin("loading the voice receive extension")
        try:
            from discord.ext import voice_recv  # noqa: F401

            ok("voice receive extension available")
        except ImportError:
            fail("discord-ext-voice-recv is missing; Snippy cannot hear anything without it.")

        if spin is not None:
            spin.begin("checking speech recognition")
        try:
            import faster_whisper  # noqa: F401

            ok("speech recognition available")
        except ImportError:
            ok(
                "speech recognition not installed; spoken triggers are off. "
                "Run ./install.sh --asr to add it."
            )

        if spin is not None:
            spin.begin("reading the config")
        config = load_file_defaults(args.config)
        ok(f"config loaded from {args.config}")
        for problem in config.validate():
            fail(f"config: {problem}")

        if spin is not None:
            spin.begin("testing the data directory")
        data_dir = Path(args.data_dir)
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            probe = data_dir / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            ok(f"data directory writable at {data_dir.resolve()}")
        except OSError as exc:
            fail(f"data directory {data_dir} is not writable: {exc}")
    finally:
        if spin is not None:
            spin.close()

    if not live:
        print(f"Snippy {__version__}\n")
        for note in notes:
            print(f"  ok    {note}")
        for message in problems:
            print(f"  FAIL  {message}")
    print()
    if problems:
        detail = f"{len(problems)} problem(s) found."
        print(f"{RED}{detail}{RESET}" if live else detail)
        return 1
    detail = "Everything checks out. Start the bot with: python -m snippy"
    print(f"{GREEN}{detail}{RESET}" if live else detail)
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
