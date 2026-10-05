"""Turning a spoken sentence into a clip request.

Transcription is the easy half. The hard half is that people say "hey snippy
clip that bit from before lunch" and expect the same thing as ``/clip`` with a
time, a duration, and a visibility. This module matches the trigger loosely,
then mines the words around it for modifiers.

Matching is deliberately generous. A clipped word, a mis-heard homophone, or
some chatter before the trigger should not stop a clip from happening, because
the failure mode of a false negative is a person saying "what?" into the void.
False positives are cheap: the result is shown as a card with buttons, so
deleting a clip nobody wanted takes one click.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..util import natime

# Words that carry no meaning here and only make phrase matching harder.
_FILLER = {
    "hey", "hi", "hello", "yo", "ok", "okay", "so", "um", "uh", "er",
    "please", "could", "you", "can", "would", "will", "there", "guys",
    "folks", "people", "now", "snippy", "snippyy", "sneaky", "snip",
}

_TRIGGER_VERBS = {"clip", "clips", "clipped", "record", "save", "capture", "grab"}
_REPLAY_VERBS = {"replay", "rewind", "play"}

# The looser fallback fires on a bare verb with no configured phrase around it.
# These are the verbs specific enough to be safe on their own; "play" and
# "save" are excluded because "let's play some board games" and "save me a
# seat" are ordinary sentences, not clip requests.
_UNAMBIGUOUS = {"clip", "replay", "rewind"}
_CONTEXT_NEEDED = {"record", "capture", "grab", "clips", "save", "play"}
# Words that make a sentence about clipping rather than merely containing a verb.
_CLIP_CONTEXT = {
    "that", "this", "it", "bit", "part", "piece", "second", "seconds",
    "minute", "minutes", "ago", "last", "past", "back", "voice", "audio",
}

# Words that look like a person's name in "clip just X" but never are.
_NOT_A_NAME = {
    "this", "that", "it", "me", "my", "mine", "us", "them", "him", "her",
    "one", "bit", "part", "the", "a", "an", "everyone", "all", "everyone's",
    "last", "next", "first", "voice", "thing", "stuff",
}

_STYLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:separat(?:e|ely|ed|es)|individually|each (?:person|one|voice)|stems?|per person)\b"), "stems"),
    (re.compile(r"\b(?:active|dead air|cut the silence|skip the quiet|quiet parts)\b"), "active"),
    (re.compile(r"\b(?:duck|background|behind)\b"), "duck"),
    (re.compile(r"\b(?:just|only) (?:my|me|i)\b|\bmy voice\b|\bonly mine\b"), "solo"),
    (re.compile(r"\b(?:everyone|all of you|everybody|the whole (?:call|conversation))\b"), "mix"),
]

_VISIBILITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:dm|dms|direct message|privately|private|in my dms?|to me only|just me)\b"), "dm"),
    (re.compile(r"\b(?:public|publicly|post it here|post here|post it|share it|in here)\b"), "public"),
]

_NUMBER = (
    r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|twenty|thirty|forty|fifty|sixty|couple)"
)
_DURATION_PATTERN = re.compile(
    rf"\b(?:last|past|previous|about|around)?\s*{_NUMBER}\s*"
    r"(second|seconds|sec|secs|minute|minutes|min|mins|m)\b"
)
_WHEN_PATTERN = re.compile(r"\b(?:from|at|around|since|back to|go back to)\s+(?P<expr>[^,.;]+)")
_INTO_PATTERN = re.compile(
    r"(?P<expr>.+?)\s+(?:in)?to\s+(?:the\s+)?(?:call|session|chat|conversation|start|beginning)"
)


@dataclass
class Intent:
    """A parsed spoken request."""

    name: str = "none"
    score: float = 0.0
    text: str = ""
    style: str | None = None
    visibility: str | None = None
    duration: float | None = None
    when: natime.TimeRef | None = None
    target: str | None = None
    matched_phrase: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return self.name != "none"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and drop filler words, for phrase matching."""
    return " ".join(w for w in _loose(text).split() if w not in _FILLER)


def _loose(text: str) -> str:
    """Lowercase and strip punctuation but keep every word.

    Modifier patterns run against this rather than :func:`normalize`, because
    "only" and "just" are exactly the words that carry meaning in "clip only my
    voice" and stripping them as filler would lose the whole request.
    """
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def similarity(needle: str, haystack: str) -> float:
    """How well a configured phrase matches something actually said.

    Three passes, most confident first. Containment catches the common case of
    chatter around a correct trigger, and the token test catches a phrase whose
    words all survived but whose order was mangled by the transcriber.
    """
    if not needle or not haystack:
        return 0.0
    if needle == haystack:
        return 1.0
    if needle in haystack:
        # Length penalty: "clip it" inside a long sentence is less convincing
        # than the same phrase said on its own.
        return 0.9 + 0.08 * (len(needle) / len(haystack))
    ratio = SequenceMatcher(None, needle, haystack).ratio()
    if ratio >= 0.75:
        return ratio
    needle_tokens = set(needle.split())
    hay_tokens = set(haystack.split())
    if needle_tokens and needle_tokens <= hay_tokens:
        return 0.7 + 0.15 * (len(needle_tokens) / len(hay_tokens))
    return ratio


def match_phrase(
    text: str, phrases: list[str], threshold: float = 0.82
) -> tuple[str, float] | None:
    """Find the best-matching configured phrase in `text`."""
    normalized = normalize(text)
    if not normalized:
        return None
    best: tuple[str, float] | None = None
    for phrase in phrases:
        score = similarity(normalize(phrase), normalized)
        if score >= threshold and (best is None or score > best[1]):
            best = (phrase, score)
    return best


def _verb_fallback(text: str) -> tuple[str, float] | None:
    """Fire on a bare trigger verb when no configured phrase matched.

    People add modifiers to the trigger constantly: "clip the last thirty
    seconds" contains none of the configured phrases but is unmistakably a clip
    request. A verb that is specific enough on its own ("clip", "rewind") fires
    immediately; a common one ("record", "save", "play") additionally needs
    something clip-shaped in the sentence, which is what keeps "save me a seat"
    and "let's play some board games" out.
    """
    tokens = _loose(text).split()
    if not tokens or len(tokens) > 8:
        return None
    present = set(tokens)
    if present & _UNAMBIGUOUS:
        return ("trigger verb", 0.7)
    if (present & _CONTEXT_NEEDED) and (present & _CLIP_CONTEXT):
        return ("trigger verb", 0.7)
    return None


def _extract_modifiers(text: str, matched: str) -> Intent:
    """Mine the sentence around a matched trigger for the rest of the request."""
    intent = Intent(text=text, matched_phrase=matched)
    loose = _loose(text)

    for pattern, style in _STYLE_PATTERNS:
        if pattern.search(loose):
            intent.style = style
            break
    for pattern, visibility in _VISIBILITY_PATTERNS:
        if pattern.search(loose):
            intent.visibility = visibility
            break

    duration = _DURATION_PATTERN.search(loose)
    if duration:
        seconds = natime.parse_duration(duration.group(0))
        if seconds:
            intent.duration = seconds

    into = _INTO_PATTERN.search(loose)
    if into:
        parsed = natime.parse_when(f"{into.group('expr')} into the call")
        if parsed is not None:
            intent.when = parsed
    else:
        when = _WHEN_PATTERN.search(loose)
        if when:
            parsed = natime.parse_when(when.group("expr").strip())
            if parsed is not None:
                intent.when = parsed
        elif intent.when is None:
            # A bare "yesterday" or "8pm" still reads as a time reference, but
            # only when what is left after removing the trigger is unambiguous.
            leftover = re.sub(
                r"\b(?:clip|record|save|capture|grab|replay|play|rewind|this|that|it|bit|part)\b",
                " ",
                loose,
            )
            leftover = " ".join(w for w in leftover.split() if w not in _FILLER and w not in _NOT_A_NAME)
            if leftover:
                candidate = natime.parse_when(leftover)
                if candidate is not None and candidate.kind in ("ago", "clock", "weekday", "offset"):
                    intent.when = candidate

    # Only an explicit "just X" or "only X" names a person. Reading the word
    # after "clip" would make "clip that" a request to solo a speaker named
    # "that", which is how a demo ends up embarrassing somebody.
    target = re.search(r"\b(?:just|only)\s+([a-z][a-z0-9_'-]{1,30})\b", loose)
    if target and target.group(1) not in _NOT_A_NAME and target.group(1) not in _TRIGGER_VERBS:
        intent.target = target.group(1)
        if intent.style is None:
            intent.style = "solo"
    return intent


def parse_intent(
    text: str,
    *,
    clip_phrases: list[str],
    replay_phrases: list[str],
    threshold: float = 0.82,
) -> Intent:
    """Classify a spoken sentence as a clip request, a replay request, or neither."""
    clip = match_phrase(text, clip_phrases, threshold)
    replay = match_phrase(text, replay_phrases, threshold)

    if clip and (not replay or clip[1] >= replay[1]):
        intent = _extract_modifiers(text, clip[0])
        intent.name = "clip"
        intent.score = clip[1]
        return intent
    if replay:
        intent = _extract_modifiers(text, replay[0])
        intent.name = "replay"
        intent.score = replay[1]
        return intent

    fallback = _verb_fallback(text)
    if fallback:
        intent = _extract_modifiers(text, fallback[0])
        tokens = set(_loose(text).split())
        intent.name = "replay" if tokens & {"replay", "rewind"} else "clip"
        intent.score = fallback[1]
        return intent
    return Intent(text=text)


def looks_like_trigger(text: str) -> bool:
    """Cheap prefilter before spending CPU on a full transcription."""
    lowered = text.lower()
    return any(word in lowered for word in _TRIGGER_VERBS | _REPLAY_VERBS)
