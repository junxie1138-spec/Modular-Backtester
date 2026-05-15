import random
from pathlib import Path
from unittest import mock

import pytest


def _seed_tree(root: Path) -> None:
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "wfo").mkdir(parents=True, exist_ok=True)
    (root / "output" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies" / "registry.py").write_text(
        "def register_strategy(cls): return cls\n", encoding="utf-8",
    )


def _valid_generation_result(strategy_id: str, summary: str = "valid test idea"):
    """Build a GenerationResult with a strategy that passes Tier 1 + Tier 2."""
    from factory.generate import GenerationResult
    fixtures = Path(__file__).parent / "fixtures"
    src = (fixtures / "valid_strategy.py").read_text(encoding="utf-8")
    cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    src = src.replace('strategy_id = "gen_test_valid"', f'strategy_id = "{strategy_id}"')
    cfg = cfg.replace("gen_test_valid", strategy_id)
    return GenerationResult(parsed={
        "strategy_id": strategy_id,
        "one_line_summary": summary,
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": src,
        "config_file": cfg,
    }, cost_usd=0.04, raw_stdout="{}")


def _stub_stage_result(stage: str):
    """Build a fake successful StageResult for the given stage."""
    from factory.stages import StageResult
    if stage == "backtest":
        parsed = {"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.05,
                  "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p"}
    elif stage == "optimize":
        parsed = {"best_params": {"size": 1.0}, "objective": "sharpe",
                  "best_score": 1.3, "run_bundle_path": "p"}
    elif stage == "wfo":
        parsed = {"oos_sharpe": 0.8, "oos_total_return": 0.1, "oos_max_drawdown": -0.05,
                  "oos_n_trades": 20, "parameter_stability": {}, "n_windows": 6,
                  "run_bundle_path": "p"}
    else:
        raise ValueError(stage)
    return StageResult(stage=stage, parsed=parsed, bundle_path=Path("p"), raw_summary={})


def test_generation_timeout_records_failed_stage_generation(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_tree(tmp_path)
    from factory.cycle import run_cycle
    from factory.generate import GenerationError
    from factory.results import read_records
    from factory.settings_loader import load_settings
    s = load_settings(tmp_settings_file)
    with mock.patch("factory.cycle.call_claude",
                    side_effect=GenerationError("claude -p timed out after 120s")):
        run_cycle(s, rng=random.Random(0))
    rec = read_records(s.paths.results_dir)[0]
    assert rec["status"] == "failed"
    assert rec["failed_stage"] == "generation"
    assert "timed out" in rec["error"]
    # No dedup entry and no strategy files.
    _dedup_shard = s.paths.dedup_dir / "local.txt"
    assert not _dedup_shard.exists() or _dedup_shard.read_text(encoding="utf-8").strip() == ""
    assert not list(s.paths.strategies_dir.glob("*.py"))


def test_validation_failure_keeps_dedup_no_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_tree(tmp_path)
    from factory.cycle import run_cycle
    from factory.generate import GenerationResult
    from factory.dedup import read_tail
    from factory.results import read_records
    from factory.settings_loader import load_settings
    s = load_settings(tmp_settings_file)
    fake = GenerationResult(parsed={
        "strategy_id": "gen_xx",
        "one_line_summary": "broken strategy",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": "not even python",
        "config_file": "run_name: gen_xx\n",
    }, cost_usd=0.01, raw_stdout="{}")
    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=42), \
         mock.patch("factory.cycle.pick_unused_strategy_id", return_value="gen_xx"):
        run_cycle(s, rng=random.Random(0))
    rec = read_records(s.paths.results_dir)[0]
    assert rec["status"] == "failed" and rec["failed_stage"] == "validation"
    # Dedup entry IS kept (§3.2).
    assert read_tail(s.paths.dedup_dir, n=10) == ["broken strategy"]
    # No files.
    assert not (s.paths.strategies_dir / "gen_xx.py").exists()


def _run_with_stage_failure(s, strategy_id, failed_stage):
    """Helper that runs a cycle with the given stage failing.

    Stubs successful stages before failed_stage; raises StageError at
    failed_stage; subsequent stages are also stubbed (but should not be
    reached).
    """
    from factory.cycle import run_cycle
    from factory.stages import StageError

    fake = _valid_generation_result(strategy_id, summary=f"trigger {failed_stage} failure")

    stage_patches = {}
    stages_order = ["backtest", "optimize", "wfo"]
    failed_idx = stages_order.index(failed_stage)
    for i, stage in enumerate(stages_order):
        target = f"factory.cycle.run_{stage}_stage"
        if i < failed_idx:
            stage_patches[target] = mock.patch(target, return_value=_stub_stage_result(stage))
        elif i == failed_idx:
            stage_patches[target] = mock.patch(
                target,
                side_effect=StageError(f"stage={stage} exit=1; traceback ...stderr tail..."),
            )
        else:
            # Stages after the failure should not be called, but stub defensively.
            stage_patches[target] = mock.patch(target, return_value=_stub_stage_result(stage))

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=42), \
         mock.patch("factory.cycle.pick_unused_strategy_id", return_value=strategy_id):
        for ctx in stage_patches.values():
            ctx.start()
        try:
            run_cycle(s, rng=random.Random(0))
        finally:
            for ctx in stage_patches.values():
                ctx.stop()


@pytest.mark.parametrize("failed_stage", ["backtest", "optimize", "wfo"])
def test_stage_failure_records_correct_failed_stage(
    tmp_settings_file: Path, tmp_path: Path, failed_stage: str,
) -> None:
    _seed_tree(tmp_path)
    from factory.results import read_records
    from factory.settings_loader import load_settings
    from factory.dedup import read_tail
    s = load_settings(tmp_settings_file)
    strategy_id = "gen_failtest"

    _run_with_stage_failure(s, strategy_id, failed_stage)

    rec = read_records(s.paths.results_dir)[0]
    assert rec["status"] == "failed"
    assert rec["failed_stage"] == failed_stage
    assert f"stage={failed_stage}" in rec["error"]
    # Dedup entry IS kept.
    assert read_tail(s.paths.dedup_dir, n=10) == [f"trigger {failed_stage} failure"]
    # The strategy file is kept (orphan accepted). The registry is not
    # edited per-strategy any more (auto-discovery replaces it).
    assert (s.paths.strategies_dir / f"{strategy_id}.py").exists()
