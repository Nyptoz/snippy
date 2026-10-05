"""The rolling on-disk archive.

The RAM rings only cover the last minute or so. The archive is what makes
``/reclip from yesterday`` and ``/replay`` possible: a continuous, already
compressed recording that rotates into fixed-length segments and is indexed so
that any past moment can be found by timestamp instead of by scanning files.

Two design constraints shape the code:

* **The audio path must never block.** The voice sink is a synchronous callback
  on the event loop, so samples go onto a bounded queue and a dedicated thread
  feeds ffmpeg. If the queue overflows, frames are dropped and counted rather
  than stalling the voice connection, because a dropped frame is far less bad
  than a stalled gateway socket.
* **The archive is a cache, not a record.** Everything here is disposable and
  bounded by retention and quota. Rendered clips are uploaded and deleted; the
  archive exists only until something is cut out of it.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("snippy.archive")

SAMPLE_RATE = 48000
CHANNELS = 2

# How much audio may sit unencoded before frames start being dropped.
QUEUE_LIMIT_FRAMES = 2500  # 50 seconds


@dataclass
class SegmentInfo:
    """A finished segment, ready to be indexed in the database."""

    session_id: int
    guild_id: int
    user_id: int | None
    start_ts: float
    duration: float
    path: str
    size_bytes: int
    runs: list[dict] = field(default_factory=list)

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.duration


class SpeechRunTracker:
    """Records who spoke when, so segments can carry a speaker timeline.

    Kept separately from the audio itself: the runs are tiny, they outlive the
    audio that has rotated away, and they are what ``/when`` draws.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: dict[int, float] = {}
        self._finished: list[dict] = []
        self._oldest_ts = time.time()

    def note(self, user_id: int, ts: float, speaking: bool) -> None:
        with self._lock:
            if speaking:
                self._current.setdefault(user_id, ts)
            elif user_id in self._current:
                self._close(user_id, ts)
            # A run that has been "speaking" for a minute is a stuck speaking
            # flag, not a monologue. Close it so the timeline stays honest.
            for uid, run_start in list(self._current.items()):
                if ts - run_start > 60.0:
                    self._close(uid, run_start + 60.0)

    def silence(self, ts: float) -> None:
        """Close every open run, because nobody is speaking any more.

        Necessary because Discord sends nothing at all from a silent user, so
        their run would never see a "stopped speaking" event and would stay
        open until the stale-run sweep sixty seconds later.
        """
        with self._lock:
            for user_id in list(self._current):
                self._close(user_id, ts)

    def _close(self, user_id: int, end_ts: float) -> None:
        start = self._current.pop(user_id, None)
        if start is None or end_ts - start < 0.25:
            return
        self._finished.append(
            {"u": user_id, "s": round(start - self._oldest_ts, 3), "d": round(end_ts - start, 3)}
        )

    def take(self, window_start: float, window_end: float) -> list[dict]:
        """Runs overlapping a window, in seconds since the session began."""
        with self._lock:
            collected = [
                run
                for run in self._finished
                if run["s"] + run["d"] >= window_start and run["s"] <= window_end
            ]
            for uid, start in self._current.items():
                if start <= window_end:
                    collected.append(
                        {
                            "u": uid,
                            "s": round(start - self._oldest_ts, 3),
                            "d": round(max(0.0, min(window_end, time.time()) - start), 3),
                        }
                    )
        return collected

    def reset(self, new_oldest_ts: float | None = None) -> None:
        with self._lock:
            self._current.clear()
            self._finished.clear()
            if new_oldest_ts is not None:
                self._oldest_ts = new_oldest_ts


class ArchiveWriter:
    """Continuously encodes a mix, or one speaker, into rotating segments."""

    def __init__(
        self,
        directory: Path,
        *,
        session_id: int,
        guild_id: int,
        user_id: int | None = None,
        bitrate: str = "32k",
        segment_seconds: float = 300.0,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        runs_provider: Callable[[float, float], list[dict]] | None = None,
        encoder: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.session_id = session_id
        self.guild_id = guild_id
        self.user_id = user_id
        self.bitrate = bitrate
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self._runs_provider = runs_provider
        self._encoder = encoder if encoder is not None else shutil.which("ffmpeg")

        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=QUEUE_LIMIT_FRAMES)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.pending: list[SegmentInfo] = []
        self.dropped_frames = 0
        self.encoded_seconds = 0.0
        self._next_index = 0

    # -- lifecycle -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether a usable encoder was found. An explicit path is verified too."""
        if self._encoder is None:
            return False
        return Path(self._encoder).exists() or shutil.which(str(self._encoder)) is not None

    def _tag(self) -> str:
        return f"u{self.user_id}" if self.user_id is not None else "mix"

    def start(self) -> None:
        if self._thread is not None or not self.available:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="snippy-archive", daemon=True)
        self._thread.start()

    def feed(self, pcm: np.ndarray) -> None:
        """Queue one frame of audio. Never blocks; drops instead if behind."""
        if self._thread is None:
            return
        data = np.ascontiguousarray(pcm, dtype=np.int16).tobytes()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self.dropped_frames += 1

    def close(self, timeout: float = 15.0) -> None:
        """Flush the queue, finalize the open segment, and stop the thread."""
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout=timeout)

    def drain_pending(self) -> list[SegmentInfo]:
        """Hand finished segments to the caller so it can index them."""
        with self._lock:
            finished = self.pending
            self.pending = []
        return finished

    # -- encoder thread ------------------------------------------------------

    def _run(self) -> None:
        state: dict = {"process": None, "path": None, "start": 0.0, "frames": 0}

        def start_segment() -> None:
            path = self.directory / f"{self.session_id}_{self._tag()}_{self._next_index:05d}.ogg"
            self._next_index += 1
            try:
                process = subprocess.Popen(
                    [
                        str(self._encoder),
                        "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-f", "s16le",
                        "-ar", str(self.sample_rate),
                        "-ac", str(self.channels),
                        "-i", "pipe:0",
                        "-c:a", "libopus",
                        "-b:a", self.bitrate,
                        "-application", "audio",
                        "-f", "ogg", "-y", str(path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:  # pragma: no cover - depends on host ffmpeg
                log.error("could not start archive encoder: %s", exc)
                path.unlink(missing_ok=True)
                return
            state.update(process=process, path=path, start=time.time(), frames=0)
            log.debug("archive segment %s opened", path.name)

        def finish_segment() -> None:
            process, path = state["process"], state["path"]
            if process is None or path is None:
                return
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
            duration = int(state["frames"]) / 50.0
            size = path.stat().st_size if path.exists() else 0
            # A quarter second is the floor for something worth indexing. It
            # has to be below the rotation size, or a segment of exactly
            # `segment_seconds` would fail its own length check and vanish.
            if size > 0 and duration >= 0.25:
                runs = (
                    self._runs_provider(state["start"], state["start"] + duration)
                    if self._runs_provider
                    else []
                )
                with self._lock:
                    self.pending.append(
                        SegmentInfo(
                            session_id=self.session_id,
                            guild_id=self.guild_id,
                            user_id=self.user_id,
                            start_ts=float(state["start"]),
                            duration=duration,
                            path=str(path),
                            size_bytes=size,
                            runs=runs,
                        )
                    )
                    self.encoded_seconds += duration
            else:
                path.unlink(missing_ok=True)
            state.update(process=None, path=None, frames=0)

        start_segment()
        limit = max(1, int(self.segment_seconds * 50))

        while True:
            item = self._queue.get()
            if item is None:
                break
            if state["process"] is None:
                start_segment()
                if state["process"] is None:
                    # No encoder available. Keep draining so feed() never
                    # blocks on a full buffer for the rest of the session.
                    continue
            try:
                process = state["process"]
                if process.stdin:
                    process.stdin.write(item)
            except (BrokenPipeError, ValueError, OSError):
                state["process"] = None
                continue
            state["frames"] += 1
            if state["frames"] >= limit:
                finish_segment()
                start_segment()

        finish_segment()
