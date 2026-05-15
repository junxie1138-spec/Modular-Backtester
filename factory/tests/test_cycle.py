import random
from pathlib import Path
from unittest import mock

import pytest

from factory.cycle import CycleOutcome, run_cycle
from factory.settings_loader import load_settings


def _seed_backtester_tree(root: Path) -> None:
    """Lay down a fake backtester tree so the cycle can write into it."""
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "wfo").mkdir(parents=True, exist_ok=True)
    (root / "output" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies" / "registry.py").write_text(
        "def register_strategy(cls): return cls\n", encoding="utf-8"
    )


def _fake_claude_result(strategy_id: str):
    from factory.generate import GenerationResult
    parsed = {
        "strategy_id": strategy_id,
        "one_line_summary": "test compression strategy",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": "# placeholder body\n",  # validation will fail this
        "config_file": "run_name: x\n",
    }
    return GenerationResult(parsed=parsed, cost_usd=0.03, raw_stdout="{}")


def test_generation_failure_writes_failed_record_and_no_dedup_entry(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)
    from factory.generate import GenerationError
    with mock.patch("factory.cycle.call_claude", side_effect=GenerationError("boom")):
        outcome = run_cycle(s, rng=random.Random(0))
    assert outcome.status == "failed"
    assert outcome.failed_stage == "generation"
    assert not s.paths.dedup_log.exists() or s.paths.dedup_log.read_text().strip() == ""
    # A failed record IS written.
    from factory.results import read_records
    records = read_records(s.paths.results_store)
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["failed_stage"] == "generation"


def test_validation_failure_writes_dedup_but_no_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)
    fake = _fake_claude_result("gen_cycle_test")
    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "failed"
    assert outcome.failed_stage == "validation"
    # Dedup log entry IS present.
    from factory.dedup import read_tail
    tail = read_tail(s.paths.dedup_log, n=10)
    assert tail == ["test compression strategy"]
    # No strategy file or config was written (validation failed before write).
    assert not (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert not (s.paths.configs_dir / "gen_1715800000.yaml").exists()
    # Registry is untouched.
    reg_text = s.paths.registry_file.read_text(encoding="utf-8")
    assert "gen_1715800000" not in reg_text


def test_complete_cycle_writes_files_registry_record(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    # Use the known-good fixture strategy as the generated body.
    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    # Rewrite ids to match the cycle-injected id.
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "valid test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    # Stub stage results.
    from factory.stages import StageResult
    bt = StageResult(stage="backtest",
                     parsed={"sharpe": 1.1, "total_return": 0.2, "max_drawdown": -0.05,
                             "win_rate": 0.6, "n_trades": 20,
                             "run_bundle_path": "p1"},
                     bundle_path=Path("p1"), raw_summary={})
    opt = StageResult(stage="optimize",
                      parsed={"best_params": {"size": 1.0}, "objective": "sharpe",
                              "best_score": 1.3, "run_bundle_path": "p2"},
                      bundle_path=Path("p2"), raw_summary={})
    wfo = StageResult(stage="wfo",
                      parsed={"oos_sharpe": 1.25, "oos_total_return": 0.18,
                              "oos_max_drawdown": -0.06, "oos_n_trades": 25,
                              "parameter_stability": {}, "n_windows": 6,
                              "run_bundle_path": "p3"},
                      bundle_path=Path("p3"), raw_summary={})

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage", return_value=bt), \
         mock.patch("factory.cycle.run_optimize_stage", return_value=opt), \
         mock.patch("factory.cycle.run_wfo_stage", return_value=wfo):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "complete"
    assert outcome.failed_stage is None
    # Strategy + config written.
    assert (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert (s.paths.configs_dir / "gen_1715800000.yaml").exists()
    # Registry has the entry.
    assert "_gen_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")
    # Record written with wfo block.
    from factory.results import read_records
    rec = read_records(s.paths.results_store)[0]
    assert rec["status"] == "complete"
    assert rec["wfo"]["oos_sharpe"] == 1.25
    # Test settings have no telegram creds, so alerted=False.
    assert rec["alerted"] is False


def test_stage_failure_writes_failed_record_keeps_dedup_and_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    from factory.stages import StageError
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "another test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage",
                    side_effect=StageError("exit=1; traceback ...")):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "failed"
    assert outcome.failed_stage == "backtest"
    # Dedup entry IS present.
    from factory.dedup import read_tail
    assert read_tail(s.paths.dedup_log, n=10) == ["another test idea"]
    # Files + registry ARE present (per §9 landmine 2: orphan accepted).
    assert (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert "_gen_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")


def test_screened_out_skips_wfo_and_promotion(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "screened test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    from factory.stages import StageResult
    bt = StageResult(stage="backtest",
                     parsed={"sharpe": 0.4, "total_return": 0.05, "max_drawdown": -0.05,
                             "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p1"},
                     bundle_path=Path("p1"), raw_summary={})
    # best_score 0.5 is below the 1.3 floor -> screen fires.
    opt = StageResult(stage="optimize",
                      parsed={"best_params": {"size": 1.0}, "objective": "sharpe",
                              "best_score": 0.5, "run_bundle_path": "p2"},
                      bundle_path=Path("p2"), raw_summary={})

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage", return_value=bt), \
         mock.patch("factory.cycle.run_optimize_stage", return_value=opt), \
         mock.patch("factory.cycle.run_wfo_stage") as run_wfo_stage:
        outcome = run_cycle(s, rng=random.Random(0))

    # WFO stage was never invoked.
    run_wfo_stage.assert_not_called()

    assert outcome.status == "complete"
    assert outcome.failed_stage is None
    assert outcome.record["screened_out"] is True
    assert outcome.record["wfo"] is None
    assert outcome.record["promotion"] is None
    assert "0.500" in outcome.record["screen_reason"]
