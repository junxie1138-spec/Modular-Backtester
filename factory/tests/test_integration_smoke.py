import random
from pathlib import Path
from unittest import mock

import pytest


@pytest.mark.slow
def test_one_full_cycle_against_real_backtester(tmp_path: Path) -> None:
    """One real cycle, end-to-end, against the real backtester.

    Uses the known-good `gen_1715800000.py` as the generated strategy body
    (with a fresh id to avoid registry collision). All three stages run as
    actual subprocesses against real CSV data in the backtester repo.
    """
    repo = Path(__file__).resolve().parents[2]   # backtester root
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text(f"""
[paths]
backtester_root  = "{repo.as_posix()}"
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_log        = "{(tmp_path / 'dedup.txt').as_posix()}"
results_dir      = "{(tmp_path / 'results').as_posix()}"
factory_log      = "{(tmp_path / 'factory.log').as_posix()}"
tmp_dir          = "{(tmp_path / '_tmp').as_posix()}"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p"]
generation_timeout_sec = 120

[stages]
stage_timeout_sec = 1800

[alerts]
alert_threshold_metric = "wfo.oos_sharpe"
alert_threshold        = 999.0
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
""", encoding="utf-8")

    from factory.settings_loader import load_settings
    s = load_settings(settings_path)

    # Use the known-good strategy as the "generated" body.
    known_src = (repo / "strategies" / "gen_1715800000.py").read_text(encoding="utf-8")
    known_cfg = (repo / "configs" / "backtests" / "gen_1715800000.yaml").read_text(encoding="utf-8")
    # Rebrand to a smoke-test id so we don't collide with the existing registry entry.
    smoke_id = "gen_factory_smoke"
    smoke_src = known_src.replace('strategy_id = "gen_1715800000"', f'strategy_id = "{smoke_id}"')
    smoke_cfg = known_cfg.replace("gen_1715800000", smoke_id)

    from factory.generate import GenerationResult
    fake_gen = GenerationResult(
        parsed={
            "strategy_id": smoke_id,
            "one_line_summary": "smoke-test range compression",
            "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
            "allow_short": False,
            "strategy_file": smoke_src,
            "config_file": smoke_cfg,
        },
        cost_usd=0.0,
        raw_stdout="{}",
    )

    from factory.cycle import run_cycle
    try:
        with mock.patch("factory.cycle.call_claude", return_value=fake_gen), \
             mock.patch("factory.cycle.pick_unused_strategy_id", return_value=smoke_id):
            outcome = run_cycle(s, rng=random.Random(0))
    finally:
        # Cleanup: remove the smoke strategy file and registry line so the
        # backtester repo isn't permanently polluted.
        strat_file = repo / "strategies" / f"{smoke_id}.py"
        cfg_file = repo / "configs" / "wfo" / f"{smoke_id}.yaml"
        if strat_file.exists():
            strat_file.unlink()
        if cfg_file.exists():
            cfg_file.unlink()
        reg = repo / "backtester" / "strategies" / "registry.py"
        text = reg.read_text(encoding="utf-8")
        cleaned = "\n".join(
            line for line in text.splitlines() if smoke_id not in line
        ) + "\n"
        reg.write_text(cleaned, encoding="utf-8")

    assert outcome.status == "complete", outcome.record
    assert outcome.record["backtest"]["sharpe"] is not None
    assert outcome.record["optimize"]["best_params"]
    assert outcome.record["wfo"]["oos_sharpe"] is not None
    assert outcome.record["wfo"]["n_windows"] > 0
