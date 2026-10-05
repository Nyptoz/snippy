"""Per-guild configuration for Snippy.

`config/default.toml` provides the baseline that ships with the install. Admins
override any subset of it at runtime through the ``/snippy config`` commands,
which persist a JSON blob per guild in SQLite. :func:`merge` layers the three
sources so that untouched settings keep following the file defaults, meaning
editing the TOML later still reaches guilds that never changed that knob.
"""

from __future__ import annotations

import dataclasses
import functools
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

STYLES = ("mix", "stems", "solo", "duet", "active", "duck")
DESTINATIONS = ("channel", "dm", "both", "here")
JOIN_POLICIES = ("always", "whitelist", "ondemand")


@dataclass
class AudioConfig:
    max_clip_seconds: float = 120.0
    default_window: float = 15.0
    pre_roll: float = 1.5
    post_roll: float = 1.5
    default_style: str = "mix"
    bitrate: str = "48k"
    silence_trim: bool = True
    normalize: bool = True
    fade_ms: int = 20

    def validate(self) -> list[str]:
        problems = []
        if not 5.0 <= self.max_clip_seconds <= 120.0:
            problems.append("audio.max_clip_seconds must be between 5 and 120")
        if not 1.0 <= self.default_window <= self.max_clip_seconds:
            problems.append("audio.default_window must be between 1 and max_clip_seconds")
        if not 0.0 <= self.pre_roll <= 10.0 or not 0.0 <= self.post_roll <= 10.0:
            problems.append("audio roll must be between 0 and 10 seconds")
        if self.default_style not in STYLES:
            problems.append(f"audio.default_style must be one of {', '.join(STYLES)}")
        if self.fade_ms < 0 or self.fade_ms > 1000:
            problems.append("audio.fade_ms must be between 0 and 1000")
        return problems


@dataclass
class RingConfig:
    mix_seconds: float = 60.0
    stem_seconds: float = 20.0
    vad_seconds: float = 120.0
    max_stem_users: int = 8
    memory_budget_mb: int = 256

    def validate(self) -> list[str]:
        problems = []
        if not 10.0 <= self.mix_seconds <= 300.0:
            problems.append("ring.mix_seconds must be between 10 and 300")
        if not 5.0 <= self.stem_seconds <= 120.0:
            problems.append("ring.stem_seconds must be between 5 and 120")
        if not 10.0 <= self.vad_seconds <= 600.0:
            problems.append("ring.vad_seconds must be between 10 and 600")
        if not 1 <= self.max_stem_users <= 32:
            problems.append("ring.max_stem_users must be between 1 and 32")
        if not 32 <= self.memory_budget_mb <= 8192:
            problems.append("ring.memory_budget_mb must be between 32 and 8192")
        return problems


@dataclass
class ArchiveConfig:
    enabled: bool = True
    retention_hours: float = 24.0
    segment_seconds: float = 300.0
    bitrate: str = "32k"
    archive_stems: bool = False
    stem_retention_hours: float = 2.0
    quota_mb: int = 5120

    def validate(self, max_retention_hours: float = 720.0) -> list[str]:
        problems = []
        if not 0.25 <= self.retention_hours <= max_retention_hours:
            problems.append(
                f"archive.retention_hours must be between 0.25 and {max_retention_hours:g}"
            )
        if not 30.0 <= self.segment_seconds <= 3600.0:
            problems.append("archive.segment_seconds must be between 30 and 3600")
        if self.archive_stems and self.stem_retention_hours > self.retention_hours:
            problems.append("archive.stem_retention_hours cannot exceed retention_hours")
        if self.quota_mb < 64:
            problems.append("archive.quota_mb must be at least 64")
        return problems


@dataclass
class TriggerConfig:
    spoken: bool = True
    phrases: list[str] = field(
        default_factory=lambda: [
            "hey snippy clip this",
            "snippy clip this",
            "clip this",
            "clip that",
            "clip it",
            "clip that bit",
            "clip that part",
            "clip me",
            "save that",
            "make that a clip",
        ]
    )
    replay_phrases: list[str] = field(
        default_factory=lambda: [
            "snippy replay that",
            "replay that",
            "play that again",
            "play that back",
            "snippy rewind",
            "rewind that",
        ]
    )
    text: bool = False
    fuzzy_threshold: float = 0.82
    min_utterance_ms: int = 400
    max_utterance_seconds: float = 6.0
    spoken_post_roll: float = 2.5

    def validate(self) -> list[str]:
        problems = []
        if not 0.5 <= self.fuzzy_threshold <= 1.0:
            problems.append("trigger.fuzzy_threshold must be between 0.5 and 1.0")
        if self.min_utterance_ms < 100:
            problems.append("trigger.min_utterance_ms must be at least 100")
        if not 0.5 <= self.max_utterance_seconds <= 15.0:
            problems.append("trigger.max_utterance_seconds must be between 0.5 and 15")
        if not self.phrases:
            problems.append("trigger.phrases must not be empty")
        return problems


@dataclass
class DeliveryConfig:
    destination: str = "channel"
    clip_channel_name: str = "snippy-clips"
    thread_per_session: bool = True
    make_public: bool = True
    mention_on_clip: bool = False

    def validate(self) -> list[str]:
        problems = []
        if self.destination not in DESTINATIONS:
            problems.append(
                f"delivery.destination must be one of {', '.join(DESTINATIONS)}"
            )
        if not self.clip_channel_name.strip():
            problems.append("delivery.clip_channel_name must not be empty")
        return problems


@dataclass
class PrivacyConfig:
    ignored_users: list[int] = field(default_factory=list)
    allowed_roles: list[int] = field(default_factory=list)
    consent_banner: bool = True
    recording_indicator: bool = True
    allow_solo: bool = True
    allow_stems: bool = True

    def validate(self) -> list[str]:
        return []


@dataclass
class JoinConfig:
    policy: str = "always"
    channels: list[int] = field(default_factory=list)
    idle_leave_minutes: float = 10.0

    def validate(self) -> list[str]:
        problems = []
        if self.policy not in JOIN_POLICIES:
            problems.append(f"join.policy must be one of {', '.join(JOIN_POLICIES)}")
        if self.policy == "whitelist" and not self.channels:
            problems.append("join.policy is 'whitelist' but join.channels is empty")
        if self.idle_leave_minutes < 0:
            problems.append("join.idle_leave_minutes must not be negative")
        return problems


@dataclass
class AsrConfig:
    enabled: bool = False
    model: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    show_transcript: bool = True
    index_search: bool = True

    def validate(self) -> list[str]:
        problems = []
        if not self.model.strip():
            problems.append("asr.model must not be empty")
        if self.device not in ("cpu", "cuda", "auto"):
            problems.append("asr.device must be cpu, cuda, or auto")
        return problems


@dataclass
class SafetyConfig:
    clips_per_hour: int = 30
    max_retention_hours: float = 720.0
    delete_after_upload: bool = True

    def validate(self) -> list[str]:
        problems = []
        if not 1 <= self.clips_per_hour <= 600:
            problems.append("safety.clips_per_hour must be between 1 and 600")
        if not 1.0 <= self.max_retention_hours <= 8760.0:
            problems.append("safety.max_retention_hours must be between 1 and 8760")
        return problems


@dataclass
class GuildConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    join: JoinConfig = field(default_factory=JoinConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> list[str]:
        problems: list[str] = []
        for section in fields(self):
            value = getattr(self, section.name)
            checker = getattr(value, "validate", None)
            if checker is None:
                continue
            if section.name == "archive":
                problems += checker(self.safety.max_retention_hours)
            else:
                problems += checker()
        return problems


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort conversion of a TOML/JSON value into the annotated type."""
    if is_dataclass(annotation) and isinstance(value, dict):
        return from_dict(annotation, value)
    if value is None:
        return None
    try:
        if annotation is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
        if annotation is str:
            return str(value)
    except (TypeError, ValueError):
        return None
    origin = getattr(annotation, "__origin__", None)
    if origin is list and isinstance(value, (list, tuple)):
        inner = annotation.__args__[0] if annotation.__args__ else None
        if inner is None:
            return list(value)
        return [_coerce(item, inner) for item in value]
    return value


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    # `from __future__ import annotations` turns field types into strings, so the
    # real annotations have to be resolved before coercion can use them.
    return get_type_hints(cls)


def from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    """Build a dataclass from a mapping, ignoring unknown keys."""
    instance = cls()
    if not data:
        return instance
    hints = _hints(cls)
    for f in fields(cls):
        if f.name not in data:
            continue
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(data[f.name], dict):
            setattr(instance, f.name, from_dict(type(current), data[f.name]))
        else:
            coerced = _coerce(data[f.name], hints.get(f.name, Any))
            if coerced is not None:
                setattr(instance, f.name, coerced)
    return instance


def merge(base: GuildConfig, overlay: dict[str, Any] | None) -> GuildConfig:
    """Return a new config with `overlay` layered on top of `base`.

    The merge goes one level deeper than a dict update on purpose. An overlay
    like ``{"audio": {"max_clip_seconds": 30}}`` describes a single change, and
    a shallow merge would replace the whole audio section and silently reset
    every other audio setting back to its hardcoded default.
    """
    data = base.to_dict()
    for section, values in (overlay or {}).items():
        existing = data.get(section)
        if isinstance(values, dict) and isinstance(existing, dict):
            data[section] = {**existing, **values}
        else:
            data[section] = values
    return from_dict(GuildConfig, data)


def diff_from(base: GuildConfig, config: GuildConfig) -> dict[str, Any]:
    """Only the sections where `config` departs from `base`.

    Persisting the difference rather than the whole config is what keeps
    `config/default.toml` a real baseline: an admin who has never touched the
    join policy still picks up a corrected default after an upgrade, instead of
    being frozen on whatever the value happened to be when they first saved
    something unrelated.
    """
    base_data = base.to_dict()
    data = config.to_dict()
    changed: dict[str, Any] = {}
    for section, values in data.items():
        baseline = base_data.get(section, {})
        if not isinstance(values, dict):
            if values != baseline:
                changed[section] = values
            continue
        section_diff = {
            key: value for key, value in values.items() if value != baseline.get(key)
        }
        if section_diff:
            changed[section] = section_diff
    return changed


def load_file_defaults(path: str | Path) -> GuildConfig:
    file_path = Path(path)
    if not file_path.is_file():
        return GuildConfig()
    with file_path.open("rb") as handle:
        return from_dict(GuildConfig, tomllib.load(handle))


def flatten(config: GuildConfig) -> dict[str, str]:
    """Render a config as dotted key/value pairs for display in the bot."""
    rows: dict[str, str] = {}

    def walk(prefix: str, node: Any) -> None:
        for f in fields(node):
            value = getattr(node, f.name)
            if is_dataclass(value):
                walk(f"{prefix}{f.name}.", value)
            else:
                rows[f"{prefix}{f.name}"] = _display(value)

    walk("", config)
    return rows


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


Style = Literal["mix", "stems", "solo", "duet", "active", "duck"]
Destination = Literal["channel", "dm", "both", "here"]
