"""The `when:` grammar, which has to survive whatever a person types."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from snippy.util import natime


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("now", "now", None),
        ("just now", "now", None),
        ("right now", "now", None),
        ("start", "start", None),
        ("the beginning", "start", None),
        ("beginning of the call", "start", None),
        ("5m ago", "ago", 300),
        ("10 minutes ago", "ago", 600),
        ("2 hours ago", "ago", 7200),
        ("1h 30m ago", "ago", 5400),
        ("3 days ago", "ago", 259200),
        ("1.5 minutes ago", "ago", 90),
        ("an hour ago", "ago", 3600),
        ("half an hour ago", "ago", 1800),
        ("in the last 30 seconds", "ago", 30),
        ("last 20 seconds", "ago", 20),
        ("the last 5 minutes", "ago", 300),
        ("2:30 into the call", "offset", 150),
        ("1:10 from the start", "offset", 70),
    ],
)
def test_relative_expressions(text, kind, value):
    ref = natime.parse_when(text)
    assert ref is not None, f"{text!r} did not parse"
    assert ref.kind == kind
    if value is not None:
        assert ref.value == pytest.approx(value)


@pytest.mark.parametrize(
    "text,minute",
    [
        ("8pm", 20 * 60),
        ("8:30pm", 20 * 60 + 30),
        ("19:45", 19 * 60 + 45),
        ("noon", 12 * 60),
        ("midnight", 0),
        ("3pm", 15 * 60),
    ],
)
def test_clock_times(text, minute):
    ref = natime.parse_when(text)
    assert ref is not None
    assert ref.kind == "clock"
    assert ref.minute == minute


def test_yesterday_shifts_the_day():
    ref = natime.parse_when("yesterday 8pm")
    assert ref is not None
    assert ref.kind == "clock"
    assert ref.day_offset == 1
    assert ref.minute == 20 * 60


def test_yesterday_at_a_plain_clock_time():
    ref = natime.parse_when("yesterday at 19:30")
    assert ref is not None
    assert (ref.kind, ref.minute) == ("clock", 19 * 60 + 30)


def test_weekday_expressions():
    ref = natime.parse_when("last friday")
    assert ref is not None
    assert ref.kind == "weekday"
    assert ref.value == 4

    with_clock = natime.parse_when("friday at 9pm")
    assert with_clock is not None
    assert (with_clock.kind, with_clock.value, with_clock.minute) == ("weekday", 4, 21 * 60)


@pytest.mark.parametrize("text", ["", "banana", "purple monkey dishwasher", "3.5.2", "the"])
def test_nonsense_is_rejected_rather_than_guessed(text):
    assert natime.parse_when(text) is None


def test_ago_resolves_against_now():
    ref = natime.parse_when("5 minutes ago")
    now = 1_700_000_000.0
    assert ref.resolve(now, None) == pytest.approx(now - 300)


def test_offset_resolves_against_the_session_start():
    ref = natime.parse_when("2:30 into the call")
    assert ref.resolve(1_700_000_000.0, 1_699_999_000.0) == pytest.approx(1_699_999_150.0)


def test_start_resolves_to_the_session_start():
    ref = natime.parse_when("start")
    assert ref.resolve(1_700_000_000.0, 1_699_999_500.0) == 1_699_999_500.0


def test_a_clock_time_earlier_today_stays_today():
    """At 9pm, '8pm' means an hour ago, not last night."""
    now = datetime(2026, 9, 28, 21, 0).timestamp()
    ref = natime.parse_when("8pm")
    resolved = datetime.fromtimestamp(ref.resolve(now, None))
    assert (resolved.hour, resolved.day) == (20, 28)


def test_a_clock_time_still_ahead_rolls_back_to_yesterday():
    """At 7pm, '8pm' has not happened yet, so it means last night."""
    now = datetime(2026, 9, 28, 19, 0).timestamp()
    ref = natime.parse_when("8pm")
    resolved = datetime.fromtimestamp(ref.resolve(now, None))
    assert (resolved.hour, resolved.day) == (20, 27)


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("90", 90),
        ("1m30s", 90),
        ("1:30", 90),
        ("1:02:03", 3723),
        ("45 seconds", 45),
        ("2 hours", 7200),
        ("1.5s", 1.5),
    ],
)
def test_durations(text, seconds):
    assert natime.parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "a while", "banana", "later maybe"])
def test_bad_durations_return_none(text):
    assert natime.parse_duration(text) is None


@pytest.mark.parametrize(
    "seconds,text",
    [(45, "45s"), (95, "1m 35s"), (3600, "1h"), (5400, "1h 30m"), (None, "unknown")],
)
def test_formatting(seconds, text):
    assert natime.format_duration(seconds) == text
