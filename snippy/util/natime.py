"""Natural-language time expressions.

Powers the ``when=`` argument of ``/reclip`` and ``/replay`` as well as the
relative-time arguments pulled out of spoken trigger phrases. Everything is
parsed into a :class:`TimeRef` and resolved against wall-clock time plus the
active session, so ``"2:30 into the call"`` and ``"yesterday at 8pm"`` are both
expressible with one grammar.

Parsing never raises on bad input: it returns ``None`` so callers can reply
with a usage hint instead of a traceback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# Rough but predictable: one "ago" expression is how humans speak.
_UNITS: dict[str, int] = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "wk": 604800, "wks": 604800, "week": 604800, "weeks": 604800,
}

_WORDS: dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "couple": 2, "few": 3, "half": 0.5,
}

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
    "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_CLOCK_WORDS = {"noon": 12 * 60, "midnight": 0}

# "5", "5.5", "two". Order matters: longer article forms come first so that
# "an hour" parses as 1 hour instead of matching "a" and leaving "n" behind.
_NUM_WORDS = (
    r"an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|couple|few|half"
)
# "5", "5.5", "two"
_NUM = rf"(\d+(?:\.\d+)?|{_NUM_WORDS})"
_UNIT = r"([a-z]+)"

_DURATION_TERM = re.compile(
    rf"{_NUM}\s*(?:of\s+)?{_UNIT}(?:s|es)?", re.IGNORECASE
)

# Phrasings the numeric grammar cannot reach, folded down to plain durations
# before parsing starts.
_PHRASE_FIXUPS: list[tuple[str, str]] = [
    (r"\bhalf\s+(?:an?\s+)?hour\b", "30 minutes"),
    (r"\bhalf\s+(?:an?\s+)?minute\b", "30 seconds"),
    (r"\bhalf\s+an?\s+second\b", "0.5 seconds"),
    (r"\ban?\s+couple\s+of\s+", "2 "),
    (r"\ban?\s+couple\s+", "2 "),
]


@dataclass(frozen=True)
class TimeRef:
    """A parsed, still-unresolved point in time.

    ``kind`` is one of:

    * ``ago``      -- ``value`` seconds before now
    * ``offset``   -- ``value`` seconds after the start of the session
    * ``clock``    -- ``minute`` minutes past midnight, ``day_offset`` days back
    * ``weekday``  -- ``value`` is a ``datetime.weekday()`` index, ``minute`` past midnight
    * ``start``    -- the beginning of the session
    * ``now``      -- this instant
    """

    kind: str
    value: float = 0.0
    day_offset: int = 0
    minute: float = 0.0
    raw: str = ""

    def resolve(self, now: float, session_start: float | None) -> float:
        """Return an absolute epoch timestamp in seconds."""
        if self.kind == "now":
            return now
        if self.kind == "start":
            return session_start if session_start is not None else now
        if self.kind == "ago":
            return now - self.value
        if self.kind == "offset":
            base = session_start if session_start is not None else now
            return base + self.value
        if self.kind == "clock":
            anchor = datetime.fromtimestamp(now) - timedelta(days=self.day_offset)
            midnight = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
            target = midnight + timedelta(minutes=self.minute)
            if target.timestamp() > now + 60:
                target -= timedelta(days=1)
            return target.timestamp()
        if self.kind == "weekday":
            anchor = datetime.fromtimestamp(now)
            delta = (anchor.weekday() - int(self.value)) % 7
            if delta == 0:
                delta = 7
            anchor = anchor - timedelta(days=delta + self.day_offset)
            midnight = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
            return (midnight + timedelta(minutes=self.minute)).timestamp()
        return now

    def describe(self) -> str:
        if self.kind == "now":
            return "right now"
        if self.kind == "start":
            return "the start of the session"
        if self.kind == "ago":
            return f"{format_duration(self.value)} ago"
        if self.kind == "offset":
            return f"{format_duration(self.value)} into the session"
        if self.kind == "clock":
            stamp = f"{int(self.minute) // 60:02d}:{int(self.minute) % 60:02d}"
            return f"{stamp} yesterday" if self.day_offset else stamp
        if self.kind == "weekday":
            return _WEEKDAY_NAMES.get(int(self.value), "that day")
        return self.raw or "that time"


_WEEKDAY_NAMES = {v: k.capitalize() for k, v in _WEEKDAYS.items() if len(k) > 3}


def _number(token: str) -> float | None:
    token = token.strip().lower()
    try:
        return float(token)
    except ValueError:
        return _WORDS.get(token)


def _duration_seconds(text: str) -> float | None:
    """Sum every ``<number><unit>`` pair in the text, e.g. ``"1h 30m"``."""
    total = 0.0
    found = False
    pos = 0
    # A manual scan rather than findall, for two reasons: compact forms like
    # "1m30s" must be able to start a term right where the previous one ended,
    # and a bogus unit should only skip one character instead of swallowing the
    # rest of the string. That second rule is what lets "half an hour" fall
    # through to "an hour" once "half an" turns out not to be a real unit.
    while pos <= len(text):
        match = _DURATION_TERM.search(text, pos)
        if not match:
            break
        number = _number(match.group(1))
        unit = match.group(2).lower().rstrip("s") or match.group(2).lower()
        if number is not None and unit in _UNITS:
            total += number * _UNITS[unit]
            found = True
            pos = match.end()
        else:
            pos = match.start() + 1
    return total if found else None


def parse_duration(text: str | None, *, bare_number_is_seconds: bool = True) -> float | None:
    """Parse a length such as ``"90"``, ``"1m30s"``, or ``"1:30"`` into seconds."""
    if text is None:
        return None
    value = text.strip().lower().replace(" ", "")
    for pattern, replacement in _PHRASE_FIXUPS:
        value = re.sub(pattern, replacement, value)
    if not value:
        return None
    if re.fullmatch(r"\d+:\d{1,2}(:\d{1,2})?", value):
        parts = [int(p) for p in value.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value) if bare_number_is_seconds else float(value) * 60
    return _duration_seconds(value)


def parse_when(text: str | None) -> TimeRef | None:
    """Parse a point-in-time expression. See :class:`TimeRef` for the grammar."""
    if not text:
        return None
    raw = text.strip()
    value = raw.strip().lower()
    if not value:
        return None
    for pattern, replacement in _PHRASE_FIXUPS:
        value = re.sub(pattern, replacement, value)
    value = value.replace("please", " ").replace("could you", " ")
    value = re.sub(r"\b(?:go|rewind|clip|replay|scrub)\s+back\s+to\b", " ", value)
    value = re.sub(r"^\s*(?:at|around|about|roughly)\s+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.?")

    if re.fullmatch(r"(now|just now|right now|this moment)", value):
        return TimeRef("now", raw=raw)
    if re.fullmatch(
        r"(?:the\s+)?(?:start|beginning)(?:\s+of\s+(?:the\s+)?(?:call|session))?", value
    ):
        return TimeRef("start", raw=raw)

    # "2:30 into the call" / "1:10 from the start"
    into = re.fullmatch(r"(.+?)\s+(?:in)?to\s+(?:the\s+)?(?:call|session|start|beginning)", value)
    if into:
        seconds = parse_duration(into.group(1))
        if seconds is not None:
            return TimeRef("offset", seconds, raw=raw)
    from_start = re.fullmatch(r"(.+?)\s+from\s+(?:the\s+)?(?:call\s+)?(?:start|beginning)", value)
    if from_start:
        seconds = parse_duration(from_start.group(1))
        if seconds is not None:
            return TimeRef("offset", seconds, raw=raw)

    # "<duration> ago" / "<duration> back"
    ago = re.fullmatch(r"(?:the\s+)?(?:last|past|previous)?\s*(.+?)\s+(?:ago|back|earlier)", value)
    if ago:
        seconds = _duration_seconds(ago.group(1)) or parse_duration(ago.group(1))
        if seconds is not None:
            return TimeRef("ago", seconds, raw=raw)
    if re.fullmatch(r"(?:in\s+)?the\s+last\s+(.+)", value):
        seconds = _duration_seconds(value) or parse_duration(value)
        if seconds is not None:
            return TimeRef("ago", seconds, raw=raw)
    if re.fullmatch(r"(?:the\s+)?last\s+(.+)", value):
        seconds = _duration_seconds(value) or parse_duration(value)
        if seconds is not None:
            return TimeRef("ago", seconds, raw=raw)

    day_offset = 0
    if re.search(r"\byesterday\b", value):
        day_offset = 1
        value = value.replace("yesterday", " ")
    elif re.search(r"\b(?:today|tonight)\b", value):
        value = re.sub(r"\b(?:today|tonight)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.?")
    if not value and day_offset:
        return TimeRef("clock", 0, day_offset, 12 * 60, raw=raw)  # yesterday lunchtime

    weekday_match = re.fullmatch(r"(?:on\s+|last\s+)?(\w+)(?:\s+at)?\s*(.*)", value)
    if weekday_match and weekday_match.group(1) in _WEEKDAYS:
        name = weekday_match.group(1)
        rest = weekday_match.group(2).strip()
        minutes = 0.0
        if rest:
            parsed = parse_clock(rest)
            if parsed is None:
                return None
            minutes = parsed
        return TimeRef("weekday", _WEEKDAYS[name], day_offset, minutes, raw=raw)

    if re.fullmatch(
        r"(?:at\s+)?(?:\d{1,2}(?::\d{2})?(?::\d{2})?\s*(?:am|pm)?|noon|midnight)", value
    ):
        minutes = parse_clock(value)
        if minutes is None:
            return None
        return TimeRef("clock", 0, day_offset, minutes, raw=raw)

    return None


def parse_clock(text: str) -> float | None:
    """Parse ``"8"``, ``"8:30"``, ``"8.30pm"``, ``"19:45"``, ``"noon"`` into minutes past midnight."""
    value = text.strip().lower().replace(" ", "")
    value = re.sub(r"^(?:at|around|about|roughly)\s*", "", value)
    if value in _CLOCK_WORDS:
        return float(_CLOCK_WORDS[value])
    value = value.replace(".", ":")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?(?::(\d{2}))?(am|pm)?", value)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if minute > 59:
        return None
    meridiem = match.group(4)
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem is None and hour <= 7:
        # People who say "snippy replay 8pm" without the meridiem mean evening.
        hour += 12
    if hour > 23:
        return None
    return float(hour * 60 + minute)


def format_duration(seconds: float | None) -> str:
    """Render a length as a compact human string such as ``1h 30m`` or ``45s``."""
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:g}s"
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    if minutes and secs:
        return f"{minutes}m {secs}s"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"
