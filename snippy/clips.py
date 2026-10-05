"""Building a clip out of a request.

A request names a window in time, a style, and who should be able to see the
result. This module turns that into a rendered file, from either source: the
in-memory rings for the recent past, or the rolling archive for anything older.

The styles differ in what they do to the samples, not in how they get delivered:

``mix``     everyone summed, the default
``solo``    one speaker, isolated
``duet``    two speakers summed
``stems``   one file per speaker plus the mix, zipped past a few files
``active``  the mix with interior dead air cut out
``duck``    the named speaker in front, everyone else pushed down
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import render
from .audio.render import safe_filename
from .voice import vad

log = logging.getLogger("snippy.clips")

STYLES = ("mix", "stems", "solo", "duet", "active", "duck")

# Discord accepts up to ten files on a message. Past this many stems a zip is
# kinder than a wall of attachments, so the threshold sits well below the limit.
MAX_ATTACHMENTS = 6


@dataclass
class ClipRequest:
    """Everything needed to produce one clip."""

    guild_id: int
    channel_id: int
    session_id: int | None
    start: float
    end: float
    style: str = "mix"
    users: list[int] = field(default_factory=list)
    requester_id: int | None = None
    caption: str | None = None
    private: bool = False
    source: str = "ram"
    # Set when rendering from the archive: the segment paths and where the
    # requested window sits inside each of them.
    segments: list[tuple[Path, float, float]] = field(default_factory=list)
    transcript: str | None = None
    speaker_names: dict[int, str] = field(default_factory=dict)
    clip_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ClipResult:
    """A rendered clip, ready to upload."""

    clip_id: str
    path: Path
    filename: str
    duration: float
    style: str
    speakers: list[int]
    extras: list[Path] = field(default_factory=list)
    manifest: str | None = None
    deleted: bool = False

    def all_files(self) -> list[Path]:
        return [self.path, *self.extras]

    def cleanup(self) -> None:
        """Remove every rendered file. Clip files are disposable by design."""
        for item in self.all_files():
            if item.exists():
                item.unlink(missing_ok=True)
        self.extras.clear()


class ClipBuilder:
    """Renders clips and keeps them around only until they are uploaded."""

    def __init__(self, data_dir: Path, config, db) -> None:
        self.workdir = Path(data_dir) / "clips"
        self.archive_dir = Path(data_dir) / "archive"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.db = db

    # -- public API ----------------------------------------------------------

    def new_clip_id(self) -> str:
        return uuid.uuid4().hex[:10]

    async def build(self, request: ClipRequest) -> ClipResult:
        """Render a request from whichever source it names."""
        config = self.config
        limit = config.audio.max_clip_seconds
        if request.duration > limit:
            request.end = request.start + limit

        if request.style == "stems":
            return await self._build_stems(request)
        if request.source == "archive":
            pcm = await self._read_archive(request)
            if pcm is None:
                raise LookupError("that moment is no longer in the archive")
        else:
            pcm = await self._read_ram(request)
        if pcm is None or pcm.size == 0:
            raise LookupError("there is no audio in that window yet")
        if is_silent(pcm):
            # Without this, a window of pure silence renders into a valid but
            # empty clip that plays as nothing, which reads as a broken bot.
            raise LookupError("that window is silent, nothing to clip")

        pcm = self._apply_style(pcm, request)
        if pcm.size == 0:
            raise LookupError("that window is silent")
        if is_silent(pcm):
            raise LookupError("that window is silent, nothing to clip")
        if config.audio.silence_trim:
            pcm = vad.trim_silence(pcm)

        clip_id = request.clip_id or self.new_clip_id()
        filename = self._filename(request)
        path = await render.render_pcm(
            pcm,
            self.workdir / f"{clip_id}{render.CLIP_SUFFIX}",
            bitrate=config.audio.bitrate,
            normalize=config.audio.normalize,
            fade_ms=config.audio.fade_ms,
        )
        return ClipResult(
            clip_id=clip_id,
            path=path,
            filename=filename,
            duration=pcm.shape[0] / render.SAMPLE_RATE,
            style=request.style,
            speakers=self._speakers_for(request, pcm),
        )

    # -- sources -------------------------------------------------------------

    async def _read_ram(self, request: ClipRequest) -> np.ndarray | None:
        sink = self._sink_for(request)
        if sink is None:
            return None
        return await self._read_ram_style(request, sink)

    async def _read_ram_style(self, request: ClipRequest, sink) -> np.ndarray | None:
        style = request.style
        if style in ("mix", "active"):
            return sink.read_mix(request.start, request.duration)
        if style == "duck":
            mix = sink.read_mix(request.start, request.duration)
            if mix is None or mix.size == 0 or not request.users:
                return mix
            primary = sink.read_stem(request.users[0], request.start, request.duration)
            if primary is None or primary.size != mix.shape[0]:
                return mix
            return _duck(mix, primary)
        users = self._resolve_users(request)
        if style == "duet" and len(users) >= 2:
            first = sink.read_stem(users[0], request.start, request.duration)
            second = sink.read_stem(users[1], request.start, request.duration)
            return _sum(first, second)
        if style == "solo" and users:
            return sink.read_stem(users[0], request.start, request.duration)
        return sink.read_mix(request.start, request.duration)

    async def _read_archive(self, request: ClipRequest) -> np.ndarray | None:
        """Decode the requested window out of the covering segments.

        Decoded to PCM rather than re-encoded directly, because a window that
        straddles two segments has to be stitched and the ``active`` style needs
        to inspect the audio before deciding what to keep.
        """
        if not request.segments:
            return None
        wanted_start, wanted_end = request.start, request.end
        pieces: list[np.ndarray] = []
        for path, seg_start, seg_end in request.segments:
            if seg_end < wanted_start or seg_start > wanted_end:
                continue
            offset = max(0.0, wanted_start - seg_start)
            length = min(seg_end, wanted_end) - max(seg_start, wanted_start)
            if length <= 0:
                continue
            block = await render.decode_to_pcm(
                path, start_offset=offset, duration=length
            )
            if block.size:
                pieces.append(block)
        if not pieces:
            return None
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    # -- styles --------------------------------------------------------------

    def _apply_style(self, pcm: np.ndarray, request: ClipRequest) -> np.ndarray:
        if request.style == "active":
            return vad.squeeze(pcm)
        return pcm

    async def _build_stems(self, request: ClipRequest) -> ClipResult:
        """One file per speaker, plus the mix, zipped once the count gets silly."""
        sink = self._sink_for(request) if request.source == "ram" else None
        files: list[tuple[str, Path]] = []
        clip_id = request.clip_id or self.new_clip_id()
        speakers: list[int] = []

        if sink is not None:
            mix = sink.read_mix(request.start, request.duration)
            if mix is not None and mix.size:
                mix_path = await render.render_pcm(
                    mix,
                    self.workdir / f"{clip_id}-mix{render.CLIP_SUFFIX}",
                    bitrate=self.config.audio.bitrate,
                    normalize=self.config.audio.normalize,
                    fade_ms=self.config.audio.fade_ms,
                )
                files.append((f"all-talkers{render.CLIP_SUFFIX}", mix_path))
            for user_id in sorted(sink.rings.known_users()):
                if request.users and user_id not in request.users:
                    continue
                stem = sink.read_stem(user_id, request.start, request.duration)
                if stem is None or stem.size == 0:
                    continue
                if self.config.audio.silence_trim:
                    stem = vad.trim_silence(stem)
                if stem.size == 0:
                    continue
                speakers.append(user_id)
                stem_path = await render.render_pcm(
                    stem,
                    self.workdir / f"{clip_id}-u{user_id}{render.CLIP_SUFFIX}",
                    bitrate=self.config.audio.bitrate,
                    normalize=self.config.audio.normalize,
                    fade_ms=self.config.audio.fade_ms,
                )
                label = request.speaker_names.get(user_id) or f"speaker-{user_id}"
                files.append((f"{safe_filename(label, f'speaker-{user_id}')}{render.CLIP_SUFFIX}", stem_path))

        if not files:
            raise LookupError("no speaker audio in that window")

        filename = f"{self._filename(request)}"
        if len(files) <= MAX_ATTACHMENTS:
            # A handful of stems is friendlier as separate attachments, each
            # individually playable, than as one zip nobody wants to open.
            main, *rest = files
            return ClipResult(
                clip_id=clip_id,
                path=main[1],
                filename=main[0],
                duration=request.duration,
                style=request.style,
                speakers=speakers,
                extras=[path for _, path in rest],
                manifest="\n".join(name for name, _ in files),
            )

        zip_name = f"{safe_filename(filename, 'stems')}-stems.zip"
        zip_path = await render.zip_stems(files, self.workdir / f"{clip_id}-stems.zip")
        for _, extra in files:
            extra.unlink(missing_ok=True)
        return ClipResult(
            clip_id=clip_id,
            path=zip_path,
            filename=zip_name,
            duration=request.duration,
            style=request.style,
            speakers=speakers,
            manifest="\n".join(name for name, _ in files),
        )

    # -- helpers -------------------------------------------------------------

    def _sink_for(self, request: ClipRequest):
        getter = getattr(self, "sink_provider", None)
        if getter is None:
            return None
        return getter(request.guild_id, request.channel_id)

    def _resolve_users(self, request: ClipRequest) -> list[int]:
        return [u for u in request.users if u]

    def _speakers_for(self, request: ClipRequest, pcm: np.ndarray) -> list[int]:
        """Who actually spoke inside the clip, not merely who was connected.

        Reading it from the speech runs rather than the ring registry matters:
        a two-hour call with a minute-long clip should credit the one person
        talking in it, not everyone who ever joined.
        """
        if request.style in ("solo", "duet", "duck"):
            return self._resolve_users(request)
        sink = self._sink_for(request)
        if sink is None:
            return []
        runs = sink.tracker.take(request.start, request.end)
        if runs:
            return sorted({int(run["u"]) for run in runs})
        return sorted(u for u in sink.rings.known_users() if u)

    def _filename(self, request: ClipRequest) -> str:
        if request.caption:
            base = safe_filename(request.caption, "clip")
        elif request.style in ("solo", "duet", "duck") and request.speaker_names:
            first = request.speaker_names.get(request.users[0]) if request.users else None
            base = safe_filename(f"{first or 'speaker'} clip", "clip")
        else:
            base = "clip"
        stamp = time.strftime("%H%M%S", time.localtime())
        return f"{base} {stamp}{render.CLIP_SUFFIX}"

    def purge_orphans(self, older_than: float = 3600.0) -> int:
        """Delete rendered files nobody uploaded, in case a render was interrupted."""
        removed = 0
        cutoff = time.time() - older_than
        if not self.workdir.exists():
            return 0
        for item in self.workdir.iterdir():
            try:
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink()
                    removed += 1
            except OSError:  # pragma: no cover
                continue
        return removed


# Below this peak level a window counts as silent rather than very quiet. Chosen
# to sit well under genuine speech while catching digital silence and the flat
# noise floor of a muted microphone.
SILENCE_FLOOR_DB = -60.0


def is_silent(pcm: np.ndarray) -> bool:
    """True when a block holds nothing worth uploading."""
    if pcm.size == 0:
        return True
    peak = float(np.max(np.abs(pcm.astype(np.float32)))) / 32768.0
    if peak <= 0.0:
        return True
    return float(vad.to_db(peak)) < SILENCE_FLOOR_DB


def _sum(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return first if second is None else second
    length = min(len(first), len(second))
    total = first[:length].astype(np.int32) + second[:length].astype(np.int32)
    info = np.iinfo(np.int16)
    return np.clip(total, info.min, info.max).astype(np.int16)


def _duck(mix: np.ndarray, primary: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Bring one speaker forward by pushing everyone else down under them."""
    length = min(len(mix), len(primary))
    mix_part = mix[:length].astype(np.float32)
    primary_part = primary[:length].astype(np.float32)
    others = mix_part - primary_part
    ducked = primary_part + others * factor
    info = np.iinfo(np.int16)
    return np.clip(ducked, info.min, info.max).astype(np.int16)
