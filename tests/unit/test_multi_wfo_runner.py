from __future__ import annotations

import pandas as pd
import pytest


def _ohlcv(closes, start="2020-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": closes, "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
         "close": closes, "volume": [1_000_000] * len(closes)},
        index=idx,
    )


class _MinStrat:
    from dataclasses import dataclass

    @dataclass(slots=True)
    class _P:
        threshold: float = 0.0
        size: float = 1.0

    strategy_id = "test_runner_stub"
    uses_multi_symbol = True
    uses_per_bar = False

    @classmethod
    def params_type(cls):
        return cls._P

    def indicators(self, data, params):
        return pd.DataFrame({"sma5": data["close"].rolling(5).mean()}, index=data.index)

    def generate_signals_for_symbol(self, *, data, indicators, params):
        sig = (data["close"] > indicators["sma5"] + params.threshold).astype(float).shift(1).fillna(0)
        return pd.DataFrame({"signal": sig, "size": params.size}, index=data.index)


def _make_engine_and_optimizer():
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
    from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
    from backtester.optimize.multi_grid_search import MultiSymbolGridSearchOptimizer
    sim = MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(sizing_mode="percent_equity", size=0.1,
                               position_cap_pct=1.0, cash_reserve_pct=0.0,
                               risk_budget_pct=1.0, sector_cap_pct=1.0),
        initial_cash=100_000.0,
        broker_factory=lambda: Broker(ExecutionConfig(initial_cash=100_000.0)),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    optimizer = MultiSymbolGridSearchOptimizer(engine=engine, objective="total_return")
    return engine, optimizer


def test_runner_executes_one_window():
    from backtester.wfo.multi_runner import MultiSymbolWFORunner
    from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter
    engine, optimizer = _make_engine_and_optimizer()
    runner = MultiSymbolWFORunner(engine=engine, optimizer=optimizer)

    splitter = MultiSymbolWFOSplitter(train_bars=60, test_bars=20, step_bars=20)
    closes = [100.0 + 0.1 * i + ((-1) ** i) * 0.5 for i in range(100)]
    data = {"AAA": _ohlcv(closes)}
    windows = list(splitter.split(data=data, aux_data={}))
    assert len(windows) >= 1

    res = runner.run_window(
        strategy_cls=_MinStrat, symbols=["AAA"], sectors={"AAA": "X"},
        window=windows[0], param_space={"threshold": [0.0, 0.5, 1.0]},
    )
    assert res.window_idx == 0
    assert res.best_params is not None
    assert len(res.oos_equity_curve) == 20


def test_runner_uses_lhs_sampling_when_requested():
    from backtester.wfo.multi_runner import MultiSymbolWFORunner
    from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter
    engine, optimizer = _make_engine_and_optimizer()
    runner = MultiSymbolWFORunner(engine=engine, optimizer=optimizer)

    splitter = MultiSymbolWFOSplitter(train_bars=60, test_bars=20, step_bars=20)
    closes = [100.0 + 0.1 * i for i in range(100)]
    data = {"AAA": _ohlcv(closes)}
    windows = list(splitter.split(data=data, aux_data={}))

    res = runner.run_window(
        strategy_cls=_MinStrat, symbols=["AAA"], sectors={"AAA": "X"},
        window=windows[0],
        param_space={"threshold": [0.0, 0.5, 1.0, 1.5, 2.0]},
        sampling="lhs", random_n=3, random_seed=0,
    )
    # Just confirm it runs end-to-end.
    assert res.best_params is not None
