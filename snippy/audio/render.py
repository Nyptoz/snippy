"""Clip rendering.

Two entry points, one idea. A clip can be cut from audio still sitting in the
RAM rings, or from a segment of the rolling on-disk archive. Both end up as a
single ffmpeg invocation, so a clip is never written to a temporary WAV and
re-read; the samples go in one end of a pipe and an Ogg Opus file comes out the
other.

Everything is normalised on the way through. Discord does no automatic gain
control worth relying on, so a quiet speaker and a loud one would otherwise
produce clips at wildly different volumes depending on who was talking.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger("snippy.render")

SAMPLE_RATE = 48000
CHANNELS = 2
CLIP_SUFFIX = ".ogg"


class FFmpegMissing(RuntimeError):
    """Raised when the ffmpeg binary is not on PATH."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    path = ffmpeg_path()
    if not path:
        raise FFmpegMissing(
            "ffmpeg is not installed. Install it (Debian/Ubuntu: apt install ffmpeg, "
            "macOS: brew install ffmpeg) and restart Snippy."
        )
    return path


def build_filters(
    *, duration: float, normalize: bool = True, fade_ms: int = 20, target_db: float = -16.0
) -> str:
    """Assemble the ffmpeg filter chain for a finished clip.

    ``loudnorm`` runs in single-pass mode. Its two-pass alternative is more
    accurate but needs a second decode of the whole clip, which is not worth it
    for speech.
    """
    chain: list[str] = []
    if normalize:
        chain.append(f"loudnorm=I={target_db:g}:TP=-1.5:LRA=11")
    fade = fade_ms / 1000.0
    if fade > 0 and duration > fade * 2:
        chain.append(f"afade=t=in:st=0:d={fade:.3f}")
        chain.append(f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}")
    return ",".join(chain)


async def run_ffmpeg(args: list[str], *, stdin_data: bytes | None = None) -> None:
    """Run ffmpeg, raising with the tail of its stderr when it fails."""
    process = await asyncio.create_subprocess_exec(
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate(stdin_data)
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        message = detail[-1] if detail else f"exit code {process.returncode}"
        raise RuntimeError(f"ffmpeg failed: {message}")


async def render_pcm(
    pcm: np.ndarray,
    destination: Path,
    *,
    bitrate: str = "48k",
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    normalize: bool = True,
    fade_ms: int = 20,
) -> Path:
    """Encode an in-memory sample block to an Ogg Opus file."""
    if pcm.ndim == 1:
        pcm = pcm.reshape(-1, 1)
    data = np.ascontiguousarray(pcm, dtype=np.int16).tobytes()
    duration = pcm.shape[0] / sample_rate
    args = [
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "pipe:0",
    ]
    filters = build_filters(duration=duration, normalize=normalize, fade_ms=fade_ms)
    if filters:
        args += ["-af", filters]
    args += ["-c:a", "libopus", "-b:a", bitrate, "-f", "ogg", "-y", str(destination)]
    await run_ffmpeg(args, stdin_data=data)
    return destination


async def render_archive_range(
    source: Path,
    destination: Path,
    *,
    start_offset: float,
    duration: float,
    bitrate: str = "48k",
    normalize: bool = True,
    fade_ms: int = 20,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> Path:
    """Re-cut a window out of an archived segment.

    ``-ss`` is placed before ``-i`` so ffmpeg seeks rather than decoding from
    the start. That is exact here because every Opus frame is independently
    decodable, which is also why segment boundaries do not click.
    """
    if start_offset > 0:
        args = ["-ss", f"{start_offset:.3f}", "-i", str(source)]
    else:
        args = ["-i", str(source)]
    filters = build_filters(duration=duration, normalize=normalize, fade_ms=fade_ms)
    if filters:
        args += ["-af", filters]
    args += [
        "-t", f"{duration:.3f}",
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "ogg",
        "-y",
        str(destination),
    ]
    await run_ffmpeg(args)
    return destination


async def decode_to_pcm(
    source: Path,
    *,
    start_offset: float = 0.0,
    duration: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> np.ndarray:
    """Decode a media file to interleaved int16 PCM.

    Used when a clip needs further editing in Python (the ``active`` style, or
    a cross-segment stitch) rather than a straight transcode.
    """
    args: list[str] = []
    if start_offset > 0:
        args += ["-ss", f"{start_offset:.3f}"]
    args += ["-i", str(source)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "pipe:1",
    ]
    process = await asyncio.create_subprocess_exec(
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    raw, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg decode failed: {detail[-1] if detail else 'unknown'}")
    usable = (len(raw) // 2) * 2
    return np.frombuffer(raw[:usable], dtype=np.int16).reshape(-1, channels)


def safe_filename(text: str, fallback: str = "clip", max_length: int = 60) -> str:
    """Make something a person can actually read in a Discord attachment list."""
    cleaned = "".join(
        char if char.isalnum() or char in " -_" else " " for char in text
    ).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return fallback
    return cleaned[:max_length].rstrip(" -_")


async def zip_stems(files: list[tuple[str, Path]], destination: Path) -> Path:
    """Bundle per-speaker stems into one archive.

    Discord's ten-file limit per message makes a separate upload per speaker
    unusable for a six-person call, so anything past a handful of stems is
    zipped instead of posted piece by piece.
    """
    loop = asyncio.get_running_loop()

    def build() -> Path:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in files:
                if path.exists():
                    archive.write(path, arcname=name)
        return destination

    return await loop.run_in_executor(None, build)
