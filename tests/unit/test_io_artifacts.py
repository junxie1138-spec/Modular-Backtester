from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtester.config.models import (
    DataConfig, ExecutionConfig, PortfolioConfig, RunConfig,
)
from backtester.core.types import BacktestResult
from backtester.io.artifacts import ArtifactWriter


def _result():
    idx = pd.bdate_range("2024-01-02", periods=5)
    eq = pd.DataFrame({"equity": [100.0, 101, 102, 103, 104],
                       "cash": [100, 0, 0, 0, 104], "position_value": [0, 101, 102, 103, 0]}, index=idx)
    trades = pd.DataFrame([{"timestamp": idx[1], "side": "buy", "qty": 1.0,
                            "price": 100.0, "commission": 0.0, "notional": 100.0}])
    positions = pd.DataFrame({"qty": [0, 1, 1, 1, 0]}, index=idx)
    return BacktestResult(summary={"total_return": 0.04}, equity_curve=eq, trades=trades, positions=positions)


def _config():
    return RunConfig(
        run_name="smoke",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )


def test_writer_creates_run_dir_with_artifacts(tmp_path: Path):
    w = ArtifactWriter(root=tmp_path, run_name="smoke", now="20240514_0114")
    run_dir = w.run_dir
    assert run_dir.parent == tmp_path
    w.write_config(_config())
    w.write_result(_result())
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "positions.csv").exists()
    assert (run_dir / "equity_curve.csv").exists()


def test_run_dir_name_format(tmp_path: Path):
    w = ArtifactWriter(root=tmp_path, run_name="foo", now="20260514_0114")
    assert w.run_dir.name == "20260514_0114_foo"
