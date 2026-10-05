"""Drawing a voice channel's recent history as an image.

``/when`` produces this so people can point at a moment instead of describing
it. "Clip the bit where I screamed" is a much better request than "clip
somewhere around 8:42", and a picture settles the argument faster than a
description does.

Deliberately plain: lanes of colour, a waveform, a time axis. Pillow rather than
matplotlib, because a clip bot should not drag in a plotting stack.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (18, 20, 26)
PANEL = (28, 31, 40)
GRID = (44, 48, 60)
TEXT = (228, 232, 240)
MUTED = (150, 157, 172)
WAVE = (108, 160, 255)
WAVE_FILL = (52, 88, 160)

LANE_HEIGHT = 22
LANE_GAP = 6
WAVE_HEIGHT = 54
PADDING = 18
HEADER = 34
AXIS = 22


@dataclass
class Speaker:
    user_id: int
    label: str


def color_for(user_id: int) -> tuple[int, int, int]:
    """A stable colour per speaker, so the same person keeps the same lane."""
    digest = hashlib.sha256(str(user_id).encode()).digest()
    # Keep saturation high and value mid so lanes stay readable on dark.
    hue = digest[0] / 255.0
    r, g, b = _hsv_to_rgb(hue, 0.62, 0.92)
    return r, g, b


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "SegoeUI.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_timeline(
    runs: list[dict],
    speakers: list[Speaker],
    *,
    window_start: float,
    window_end: float,
    levels=None,
    width: int = 900,
    title: str | None = None,
) -> io.BytesIO:
    """Draw who spoke when across a window, with the mix waveform beneath."""
    duration = max(0.001, window_end - window_start)
    lanes = max(1, len(speakers))
    height = (
        PADDING * 2
        + HEADER
        + lanes * (LANE_HEIGHT + LANE_GAP)
        + WAVE_HEIGHT
        + AXIS
    )
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    label_font = _font(12)
    small_font = _font(11)
    title_font = _font(14)

    chart_left = PADDING + 118
    chart_right = width - PADDING
    chart_width = chart_right - chart_left

    heading = title or "Voice activity"
    draw.text((PADDING, PADDING), heading, fill=TEXT, font=title_font)

    def to_x(offset: float) -> float:
        clamped = (offset - window_start) / duration
        return chart_left + max(0.0, min(1.0, clamped)) * chart_width

    # Time grid. Pick a human step that gives a handful of labels rather than
    # the one or none a fixed interval would produce on a short window.
    step = _tick_step(duration)
    first = math.ceil(window_start / step) * step
    tick = first
    while tick <= window_end:
        x = to_x(tick)
        draw.line([(x, PADDING + HEADER - 6), (x, height - PADDING - AXIS)], fill=GRID)
        stamp = _clock(tick)
        draw.text((x + 3, height - PADDING - AXIS + 4), stamp, fill=MUTED, font=small_font)
        tick += step

    lane_top = PADDING + HEADER
    for index, speaker in enumerate(speakers):
        top = lane_top + index * (LANE_HEIGHT + LANE_GAP)
        draw.rounded_rectangle(
            [chart_left, top, chart_right, top + LANE_HEIGHT], radius=5, fill=PANEL
        )
        name = speaker.label
        while draw.textlength(name, font=label_font) > 108 and len(name) > 4:
            name = name[:-2]
        if name != speaker.label:
            name = name[:-1] + "…"
        draw.text((PADDING, top + 4), name, fill=TEXT, font=label_font)

    # Speech runs, drawn per lane.
    lane_of = {speaker.user_id: index for index, speaker in enumerate(speakers)}
    for run in runs:
        user_id = int(run.get("u", 0))
        lane = lane_of.get(user_id)
        if lane is None:
            continue
        start = window_start + float(run.get("s", 0.0))
        end = start + float(run.get("d", 0.0))
        x0, x1 = to_x(start), to_x(end)
        if x1 - x0 < 1:
            x1 = x0 + 1
        top = lane_top + lane * (LANE_HEIGHT + LANE_GAP) + 3
        draw.rounded_rectangle(
            [x0, top, x1, top + LANE_HEIGHT - 6], radius=4, fill=color_for(user_id)
        )

    # Mixed waveform under the lanes.
    wave_top = lane_top + lanes * (LANE_HEIGHT + LANE_GAP) + 4
    draw.rounded_rectangle(
        [chart_left, wave_top, chart_right, wave_top + WAVE_HEIGHT], radius=6, fill=PANEL
    )
    if levels is not None and len(levels):
        buckets = max(1, int(chart_width))
        values = _resample(_normalize(levels), buckets)
        # Levels are RMS amplitudes; clamp so one loud moment cannot paint the
        # entire panel and hide the shape of everything else.
        values = [max(0.0, min(1.0, v)) for v in values]
        baseline = wave_top + WAVE_HEIGHT / 2
        for index, value in enumerate(values):
            x = chart_left + index
            amplitude = max(1.0, value * (WAVE_HEIGHT / 2 - 4))
            draw.line(
                [(x, baseline - amplitude), (x, baseline + amplitude)],
                fill=WAVE if value > 0.05 else WAVE_FILL,
                width=1,
            )
    else:
        draw.text(
            (chart_left + 8, wave_top + WAVE_HEIGHT / 2 - 7),
            "no level data for this window",
            fill=MUTED,
            font=small_font,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer


def _resample(values, buckets: int):
    data = [float(v) for v in values]
    if not data:
        return [0.0] * buckets
    if len(data) == buckets:
        return data
    out = []
    step = len(data) / buckets
    for index in range(buckets):
        lo = int(index * step)
        hi = max(lo + 1, int((index + 1) * step))
        out.append(max(data[lo:hi]))
    return out


def _normalize(values):
    """Scale a level track to 0..1 against its own peak."""
    data = [float(v) for v in values]
    peak = max((abs(v) for v in data), default=0.0)
    if peak <= 0:
        return data
    return [v / peak for v in data]


_TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def _tick_step(duration: float) -> float:
    """Smallest human interval that keeps the label count in a readable range."""
    for step in _TICK_STEPS:
        if duration / step <= 7:
            return float(step)
    return float(_TICK_STEPS[-1])


def _clock(offset: float) -> str:
    total = int(offset)
    return f"{total // 60}:{total % 60:02d}" if total >= 60 else f"{total}s"
