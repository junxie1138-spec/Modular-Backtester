from pathlib import Path

from factory.settings_loader import load_settings


def test_loads_all_sections(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    assert s.paths.backtester_root.is_absolute()
    assert s.paths.strategies_dir.name == "strategies"
    assert s.paths.registry_file.parts[-2:] == ("strategies", "registry.py")
    assert s.generation.claude_cmd == "claude"
    assert "--bare" in s.generation.claude_flags
    assert s.generation.generation_timeout_sec == 60
    assert s.stages.stage_timeout_sec == 300
    assert s.alerts.alert_threshold_metric == "wfo.oos_sharpe"
    assert s.alerts.alert_threshold == 1.0
    assert s.loop.mode == "continuous"
    assert s.loop.max_cycles == 1
    assert s.dashboard.port == 8787


def test_paths_resolve_under_root(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    root = s.paths.backtester_root
    assert s.paths.strategies_dir.is_relative_to(root)
    assert s.paths.results_store.is_relative_to(root)
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
