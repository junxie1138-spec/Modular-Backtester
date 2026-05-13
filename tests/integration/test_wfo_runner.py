from __future__ import annotations

import pandas as pd

from backtester.config.models import (
    DataConfig, ExecutionConfig, OptimizationConfig, PortfolioConfig,
    RunConfig, WFOConfig,
)
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.wfo.runner import WalkForwardRunner
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher
from strategies.sma_cross import SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def _config():
    return RunConfig(
        run_name="wfo_smoke",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SYN"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(commission_bps=0.0, slippage_bps=0.0),
        portfolio=PortfolioConfig(size=1.0),
        optimization=OptimizationConfig(objective="sharpe", param_space={"fast": [5, 10], "slow": [20, 50]}),
        wfo=WFOConfig(enabled=True, train_bars=200, test_bars=50, step_bars=50),
    )


def test_wfo_runner_produces_window_results_and_stitched_output():
    data = make_ohlcv(n=500, seed=33, drift=0.0005)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective="sharpe")

    runner = WalkForwardRunner(
        engine=engine, optimizer=optimizer,
        splitter=WalkForwardSplitter(), stitcher=WalkForwardStitcher(),
    )

    out = runner.run(strategy_cls=SMACrossStrategy, full_data=data, base_config=_config())

    assert "oos_equity_curve" in out
    assert "window_results" in out
    assert len(out["window_results"]) >= 5
    for wr in out["window_results"]:
        for k in ("train_start", "train_end", "test_start", "test_end", "best_params", "train_summary", "test_summary"):
            assert k in wr
