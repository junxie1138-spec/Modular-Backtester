import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from factory.stages import (
    StageError,
    StageResult,
    build_stage_config,
    run_backtest_stage,
    run_optimize_stage,
    run_wfo_stage,
)


def _canonical_config(strategy_id: str) -> str:
    cfg = {
        "run_name": strategy_id,
        "strategy": strategy_id,
        "strategy_params": {"size": 1.0},
        "data": {"symbols": ["SPY"], "timeframe": "1d", "start": "2015-01-02",
                 "end": "2024-12-31", "source": "csv", "root": "data/raw"},
        "execution": {"initial_cash": 100000, "commission_bps": 2,
                      "slippage_bps": 5, "allow_fractional": False, "allow_short": False},
        "portfolio": {"sizing_mode": "percent_equity", "size": 0.95},
        "optimization": {"objective": "sharpe", "param_space": {"size": [0.5, 1.0]}},
        "wfo": {"enabled": True, "train_bars": 756, "test_bars": 252, "step_bars": 252},
    }
    return yaml.safe_dump(cfg, sort_keys=False)


def test_build_stage_config_rewrites_run_name(tmp_path: Path) -> None:
    canonical = tmp_path / "configs" / "wfo" / "gen_42.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")

    tmp_dir = tmp_path / "_tmp"
    bt_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="backtest",
        tmp_dir=tmp_dir,
    )
    opt_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="optimize",
        tmp_dir=tmp_dir,
    )
    wfo_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="wfo",
        tmp_dir=tmp_dir,
    )
    assert bt_cfg.exists() and opt_cfg.exists() and wfo_cfg.exists()
    assert bt_cfg != opt_cfg != wfo_cfg

    bt = yaml.safe_load(bt_cfg.read_text())
    opt = yaml.safe_load(opt_cfg.read_text())
    wfo = yaml.safe_load(wfo_cfg.read_text())
    assert bt["run_name"] == "gen_42"
    assert opt["run_name"] == "gen_42_grid"
    assert wfo["run_name"] == "gen_42_wfo"
    # Everything else carries through unchanged.
    assert bt["strategy"] == opt["strategy"] == wfo["strategy"] == "gen_42"


def test_run_backtest_stage_invokes_subprocess_and_parses(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "total_return": 0.1, "sharpe": 1.2, "max_drawdown": -0.05,
        "win_rate": 0.6, "n_trades": 20,
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc) as run_mock:
        result = run_backtest_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert run_mock.called
    cmd = run_mock.call_args[0][0]
    # Subprocess is invoked with the runner module and --config pointing at
    # the stage-specific YAML, not the canonical one.
    assert "backtester.runners.run_backtest" in cmd
    assert "--config" in cmd
    assert isinstance(result, StageResult)
    assert result.parsed["sharpe"] == pytest.approx(1.2)
    assert result.parsed["run_bundle_path"].endswith("20260101_0900_gen_42")


def test_run_optimize_stage_uses_grid_suffix(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42_grid"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "best_params": {"size": 1.0}, "best_score_objective": "sharpe",
        "best_summary": {"sharpe": 1.5, "params": {"size": 1.0}},
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        result = run_optimize_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert result.parsed["best_score"] == pytest.approx(1.5)
    assert result.parsed["run_bundle_path"].endswith("_gen_42_grid")


def test_run_wfo_stage_uses_wfo_suffix(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42_wfo"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "oos_summary": {"sharpe": 1.1, "sortino": 1.4, "total_return": 0.2, "max_drawdown": -0.06, "n_trades": 30},
        "parameter_stability": {},
        "n_windows": 6,
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        result = run_wfo_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert result.parsed["oos_sharpe"] == pytest.approx(1.1)
    assert result.parsed["oos_sortino"] == pytest.approx(1.4)
    assert result.parsed["n_windows"] == 6


def test_stage_nonzero_exit_raises_with_stderr_tail(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    fake_proc = mock.Mock(returncode=1, stdout="", stderr="boom traceback line 1\nline 2")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        with pytest.raises(StageError) as exc:
            run_backtest_stage(
                canonical_config=canonical,
                strategy_id="gen_42",
                output_runs_dir=tmp_path / "output" / "runs",
                tmp_dir=tmp_path / "_tmp",
                timeout_sec=30,
            )
    assert "boom" in str(exc.value)


def test_stage_missing_summary_raises(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    (output_runs / "20260101_0900_gen_42").mkdir(parents=True)  # bundle exists, no summary.json
    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        with pytest.raises(StageError) as exc:
            run_backtest_stage(
                canonical_config=canonical,
                strategy_id="gen_42",
                output_runs_dir=output_runs,
                tmp_dir=tmp_path / "_tmp",
                timeout_sec=30,
            )
    assert "summary" in str(exc.value).lower()
