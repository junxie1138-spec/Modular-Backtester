import pytest
from pathlib import Path

from factory.settings_loader import load_settings


def test_loads_all_sections(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    assert s.paths.backtester_root.is_absolute()
    assert s.paths.strategies_dir.name == "strategies"
    assert s.paths.registry_file.parts[-2:] == ("strategies", "registry.py")
    assert s.generation.provider == "claude"
    assert s.generation.cmd == "claude"
    assert "--bare" in s.generation.flags
    assert s.generation.claude_cmd == "claude"
    assert "--bare" in s.generation.claude_flags
    assert s.generation.generation_timeout_sec == 60
    assert s.stages.stage_timeout_sec == 300
    assert s.alerts.alert_threshold_metric == "wfo.oos_sharpe"
    assert s.alerts.alert_threshold == 1.0
    assert s.loop.mode == "continuous"
    assert s.loop.max_cycles == 1
    assert s.dashboard.port == 8787
    assert s.node_id == "local"
    assert s.sync.enabled is False


def test_paths_resolve_under_root(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    root = s.paths.backtester_root
    assert s.paths.strategies_dir.is_relative_to(root)
    assert s.paths.results_dir.is_relative_to(root)
    assert s.paths.tmp_dir.is_relative_to(root)


def test_settings_local_overrides_base(tmp_settings_file: Path) -> None:
    """A sibling settings.local.toml must override base values per section.

    This is how secrets (Telegram token) stay out of version control: the
    tracked settings.toml carries empty placeholders, the gitignored
    settings.local.toml carries the real values.
    """
    base = load_settings(tmp_settings_file)
    assert base.alerts.telegram_bot_token == ""  # placeholder in the fixture

    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text(
        '[alerts]\n'
        'telegram_bot_token = "SECRET-TOKEN-123"\n'
        'telegram_chat_id   = "-999"\n',
        encoding="utf-8",
    )
    merged = load_settings(tmp_settings_file)
    assert merged.alerts.telegram_bot_token == "SECRET-TOKEN-123"
    assert merged.alerts.telegram_chat_id == "-999"
    # Unrelated keys in the same section are preserved from the base.
    assert merged.alerts.alert_threshold_metric == "wfo.oos_sharpe"
    # Sections not mentioned in the local file are untouched.
    assert merged.dashboard.port == 8787


def test_generation_provider_fields_override_legacy_aliases(tmp_settings_file: Path) -> None:
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text(
        "[generation]\n"
        "provider = \"codex\"\n"
        "cmd = \"codex\"\n"
        "flags = [\"exec\", \"-\"]\n",
        encoding="utf-8",
    )
    s = load_settings(tmp_settings_file)
    assert s.generation.provider == "codex"
    assert s.generation.cmd == "codex"
    assert s.generation.flags == ("exec", "-")
    # Legacy aliases remain populated for older code/tests.
    assert s.generation.claude_cmd == "claude"
    assert "--bare" in s.generation.claude_flags


def test_node_id_defaults_to_local(tmp_settings_file: Path) -> None:
    """When no node_id is set anywhere, it defaults to 'local'."""
    s = load_settings(tmp_settings_file)
    assert s.node_id == "local"


def test_node_id_read_from_local_override(tmp_settings_file: Path) -> None:
    """A top-level node_id in settings.local.toml is picked up."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text('node_id = "desk"\n', encoding="utf-8")
    s = load_settings(tmp_settings_file)
    assert s.node_id == "desk"


@pytest.mark.parametrize("bad", ["Bad_ID", "-desk", "", "node id", "UPPER"])
def test_malformed_node_id_is_fatal(tmp_settings_file: Path, bad: str) -> None:
    """A node_id that is not ^[a-z0-9][a-z0-9-]*$ fails settings load."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text(f'node_id = "{bad}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="node_id"):
        load_settings(tmp_settings_file)


def test_sync_section_defaults(tmp_settings_file: Path) -> None:
    """With no [sync] section, sync is disabled with documented defaults."""
    s = load_settings(tmp_settings_file)
    assert s.sync.enabled is False
    assert s.sync.branch == "factory-pool"
    assert s.sync.remote == "origin"
    assert s.sync.push_retries == 5


def test_sync_section_explicit(tmp_settings_file: Path) -> None:
    """An explicit [sync] section overrides the defaults."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text(
        "[sync]\n"
        "enabled = true\n"
        "branch = \"pool-x\"\n"
        "push_retries = 9\n",
        encoding="utf-8",
    )
    s = load_settings(tmp_settings_file)
    assert s.sync.enabled is True
    assert s.sync.branch == "pool-x"
    assert s.sync.remote == "origin"   # untouched key keeps its default
    assert s.sync.push_retries == 9
