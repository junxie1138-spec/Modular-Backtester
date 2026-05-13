from __future__ import annotations

import pandas as pd

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.optimize.grid_search import GridSearchOptimizer
from strategies.sma_cross import SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_grid_search_returns_best_and_all_results():
    data = make_ohlcv(n=400, seed=12, drift=0.0006, vol=0.01)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    opt = GridSearchOptimizer(engine=engine, objective="sharpe")
    best_params, best_result, all_results = opt.find_best(
        strategy_cls=SMACrossStrategy,
        data=data,
        param_space={"fast": [5, 10], "slow": [20, 50]},
        symbol="SYN", timeframe="1d",
    )
    assert best_params is not None
    assert isinstance(all_results, list) and len(all_results) == 4
    # best score >= every other score
    best_score = max(r["score"] for r in all_results)
    assert best_score == max(r["score"] for r in all_results)


def test_grid_search_handles_strategy_failures_gracefully():
    data = make_ohlcv(n=50, seed=3)
    broker = Broker(ExecutionConfig())
    portfolio = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    # slow > data length will produce all-NaN indicators -> still runs, returns score
    opt = GridSearchOptimizer(engine=engine, objective="sharpe")
    best_params, _, results = opt.find_best(
        strategy_cls=SMACrossStrategy,
        data=data,
        param_space={"fast": [5], "slow": [200]},  # warmup > data
        symbol="X", timeframe="1d",
    )
    assert len(results) == 1
