"""Configuration: layering, diffing, and validation."""

from __future__ import annotations

import pytest

from snippy.config import (
    GuildConfig,
    diff_from,
    flatten,
    from_dict,
    load_file_defaults,
    merge,
)


def test_overlay_keeps_siblings_within_a_section():
    """A one-setting change must not reset the rest of that section."""
    base = GuildConfig()
    base.audio.bitrate = "96k"
    base.audio.normalize = False
    merged = merge(base, {"audio": {"max_clip_seconds": 30.0}})
    assert merged.audio.max_clip_seconds == 30.0
    assert merged.audio.bitrate == "96k"
    assert merged.audio.normalize is False


def test_merge_does_not_mutate_its_base():
    base = GuildConfig()
    merged = merge(base, {"audio": {"max_clip_seconds": 30.0}})
    merged.audio.default_window = 99.0
    assert base.audio.max_clip_seconds == 120.0
    assert base.audio.default_window == 15.0


def test_diff_reports_only_real_changes():
    base = GuildConfig()
    config = GuildConfig()
    config.audio.max_clip_seconds = 30.0
    config.delivery.destination = "dm"
    assert diff_from(base, config) == {
        "audio": {"max_clip_seconds": 30.0},
        "delivery": {"destination": "dm"},
    }


def test_diff_of_an_untouched_config_is_empty():
    assert diff_from(GuildConfig(), GuildConfig()) == {}


def test_round_trip_through_a_diff_preserves_the_result():
    base = GuildConfig()
    base.audio.bitrate = "96k"
    config = merge(base, {"audio": {"max_clip_seconds": 30.0}, "join": {"policy": "ondemand"}})
    assert merge(GuildConfig(), diff_from(GuildConfig(), config)).audio.max_clip_seconds == 30.0


def test_unknown_keys_are_ignored_rather_than_crashing():
    config = from_dict(GuildConfig, {"audio": {"nonsense": 1, "max_clip_seconds": 45.0}})
    assert config.audio.max_clip_seconds == 45.0


def test_strings_are_coerced_to_numbers():
    config = from_dict(GuildConfig, {"audio": {"max_clip_seconds": "45"}})
    assert config.audio.max_clip_seconds == 45.0


def test_a_valid_config_reports_no_problems():
    assert GuildConfig().validate() == []


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda c: setattr(c.audio, "max_clip_seconds", 500.0), "max_clip_seconds"),
        (lambda c: setattr(c.audio, "default_style", "bogus"), "default_style"),
        (lambda c: setattr(c.ring, "max_stem_users", 0), "max_stem_users"),
        (lambda c: setattr(c.delivery, "destination", "carrier-pigeon"), "destination"),
        (lambda c: setattr(c.join, "policy", "sometimes"), "policy"),
        (lambda c: setattr(c.trigger, "fuzzy_threshold", 2.0), "fuzzy_threshold"),
        (lambda c: setattr(c.trigger, "phrases", []), "phrases"),
        (lambda c: setattr(c.safety, "clips_per_hour", 0), "clips_per_hour"),
    ],
)
def test_out_of_range_values_are_reported(mutate, fragment):
    config = GuildConfig()
    mutate(config)
    problems = config.validate()
    assert any(fragment in problem for problem in problems), problems


def test_whitelist_policy_requires_channels():
    config = GuildConfig()
    config.join.policy = "whitelist"
    assert any("channels" in problem for problem in config.validate())
    config.join.channels = [1]
    assert config.validate() == []


def test_retention_is_capped_by_the_safety_ceiling():
    config = GuildConfig()
    config.archive.retention_hours = 5000.0
    assert any("retention_hours" in problem for problem in config.validate())
    config.safety.max_retention_hours = 8000.0
    assert config.validate() == []


def test_flatten_renders_every_setting():
    rows = flatten(GuildConfig())
    assert "audio.max_clip_seconds" in rows
    assert rows["audio.normalize"] == "on"
    assert rows["privacy.ignored_users"] == "none"


def test_shipped_defaults_file_loads_and_is_valid():
    config = load_file_defaults("config/default.toml")
    assert config.validate() == []
    assert config.audio.max_clip_seconds == 120.0
    assert config.join.policy == "always"
    assert "clip that" in config.trigger.phrases
    assert config.asr.enabled is False


def test_missing_defaults_file_falls_back_to_dataclass_defaults(tmp_path):
    config = load_file_defaults(tmp_path / "nope.toml")
    assert config.audio.max_clip_seconds == 120.0
