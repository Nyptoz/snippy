"""Spoken intent parsing, including the sentences that should not match."""

from __future__ import annotations

import pytest

from snippy.config import TriggerConfig
from snippy.intent.matcher import (
    looks_like_trigger,
    match_phrase,
    normalize,
    parse_intent,
    similarity,
)

CONFIG = TriggerConfig()
CLIPS = CONFIG.phrases
REPLAYS = CONFIG.replay_phrases


def run(text: str):
    return parse_intent(
        text, clip_phrases=CLIPS, replay_phrases=REPLAYS, threshold=CONFIG.fuzzy_threshold
    )


def test_normalize_strips_punctuation_and_filler():
    assert normalize("Hey Snippy, could you please clip that?!") == "clip that"


def test_similarity_ranks_a_real_match_first():
    assert similarity("clip that", "clip that") == 1.0
    assert similarity("clip that", "guys clip that now") > 0.9
    assert similarity("clip that", "the weather is nice") < 0.5


@pytest.mark.parametrize("text", [phrase for phrase in CLIPS])
def test_every_configured_phrase_matches_itself(text):
    assert run(text).name == "clip"


@pytest.mark.parametrize("text", [phrase for phrase in REPLAYS])
def test_every_replay_phrase_matches_itself(text):
    assert run(text).name == "replay"


@pytest.mark.parametrize(
    "text",
    [
        "the weather is nice today",
        "I was thinking about that",
        "let us play some board games",
        "save me a seat",
        "I recorded a video earlier",
        "grab the good snacks",
        "",
    ],
)
def test_ordinary_conversation_does_not_fire(text):
    assert run(text).name == "none"


def test_a_bare_trigger_with_a_modifier_still_fires():
    intent = run("clip the last 45 seconds")
    assert intent.name == "clip"
    assert intent.duration == pytest.approx(45.0)


def test_spelled_out_numbers_work():
    assert run("clip the last thirty seconds").duration == pytest.approx(30.0)
    assert run("clip the last two minutes").duration == pytest.approx(120.0)


def test_plain_triggers_carry_no_style():
    """'clip that' must not be read as 'solo a speaker called that'."""
    for text in ("clip that", "clip that bit", "clip it", "clip this"):
        intent = run(text)
        assert intent.style is None
        assert intent.target is None


def test_explicit_solo_phrases():
    assert run("clip only my voice").style == "solo"
    assert run("clip just ken's bit").target == "ken's"


@pytest.mark.parametrize(
    "text,style",
    [
        ("clip everyone separately", "stems"),
        ("clip each person", "stems"),
        ("clip that separately", "stems"),
        ("clip the active parts", "active"),
        ("clip that but duck the rest", "duck"),
        ("clip everyone", "mix"),
    ],
)
def test_style_phrases(text, style):
    assert run(text).style == style


@pytest.mark.parametrize(
    "text,visibility",
    [
        ("clip this and make it public", "public"),
        ("clip that but dm me", "dm"),
        ("clip that privately", "dm"),
    ],
)
def test_visibility_phrases(text, visibility):
    assert run(text).visibility == visibility


@pytest.mark.parametrize("text", ["dm me", "make it public"])
def test_visibility_alone_is_not_a_request(text):
    """A visibility word with no trigger is just someone talking."""
    assert run(text).name == "none"


def test_time_arguments_are_extracted():
    intent = run("clip that from yesterday at 8pm")
    assert intent.when is not None
    assert intent.when.kind == "clock"
    assert intent.when.minute == 20 * 60

    offset = run("clip that 2 minutes into the call")
    assert offset.when is not None
    assert offset.when.kind == "offset"
    assert offset.when.value == pytest.approx(120.0)


def test_match_phrase_finds_the_best_candidate():
    best = match_phrase("hey snippy clip that bit", CLIPS, 0.8)
    assert best is not None
    assert best[0] in CLIPS


def test_match_phrase_returns_none_for_unrelated_text():
    assert match_phrase("what a lovely afternoon", CLIPS, 0.82) is None


def test_looks_like_trigger_is_a_cheap_prefilter():
    assert looks_like_trigger("hey snippy clip this")
    assert looks_like_trigger("play that back")
    assert not looks_like_trigger("what a lovely afternoon")


def test_replay_only_fires_on_an_unambiguous_verb():
    assert run("rewind that").name == "replay"
    assert run("replay the last bit").name == "replay"
