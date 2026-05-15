import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_settings_file(tmp_path: Path) -> Path:
    """Write a complete settings.toml into tmp_path with backtester_root=tmp_path."""
    toml = textwrap.dedent(
        f"""
        [paths]
        backtester_root  = "{tmp_path.as_posix()}"
        strategies_dir   = "strategies"
        configs_dir      = "configs/wfo"
        registry_file    = "backtester/strategies/registry.py"
        output_runs_dir  = "output/runs"
        dedup_log        = "factory/data/dedup_log.txt"
        results_store    = "factory/data/results.json"
        factory_log      = "factory/logs/factory.log"
        tmp_dir          = "factory/data/_tmp"

        [generation]
        claude_cmd             = "claude"
        claude_flags           = ["-p", "--bare", "--output-format", "json"]
        generation_timeout_sec = 60

        [stages]
        stage_timeout_sec = 300

        [alerts]
        alert_threshold_metric = "wfo.oos_sharpe"
        alert_threshold        = 1.0
        telegram_bot_token     = ""
        telegram_chat_id       = ""
        dashboard_base_url     = "http://127.0.0.1:8787"

        [loop]
        mode                  = "continuous"
        inter_cycle_sleep_sec = 0
        max_cycles            = 1

        [dashboard]
        host             = "127.0.0.1"
        port             = 8787
        auto_refresh_sec = 10
        """
    ).strip()
    p = tmp_path / "settings.toml"
    p.write_text(toml, encoding="utf-8")
    return p
