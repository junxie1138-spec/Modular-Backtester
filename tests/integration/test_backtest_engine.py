from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.core.types import SignalFrame, StrategyContext
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.strategies.base import BaseStrategy
from tests.fixtures.synthetic import make_ohlcv


@dataclass(slots=True)
class _BHParams:
    size: float = 1.0


class _BuyAndHoldStrategy(BaseStrategy[_BHParams]):
    strategy_id = "_buy_and_hold_test"

    @classmethod
    def params_type(cls):
        return _BHParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params: _BHParams):
        df = pd.DataFrame(index=data.index)
        df["signal"] = 1
        df["signal"].iloc[0] = 0
        df["size"] = params.size
        return SignalFrame(data=df)


def test_engine_runs_end_to_end_and_returns_result():
    data = make_ohlcv(n=100, seed=3, drift=0.001, vol=0.005)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    result = engine.run(_BuyAndHoldStrategy(), data, _BHParams(), symbol="SYN", timeframe="1d")

    assert "total_return" in result.summary
    assert len(result.equity_curve) == 100
    assert (result.equity_curve["equity"] > 0).all()
    assert "params" in result.summary or "params" in getattr(result, "summary", {})


def test_engine_validates_data():
    broker = Broker(ExecutionConfig())
    portfolio = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    bad = make_ohlcv(n=10, seed=1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        engine.run(_BuyAndHoldStrategy(), bad, _BHParams(), symbol="X", timeframe="1d")


def _downtrending_ohlcv(n: int = 200) -> pd.DataFrame:
    """Deterministic monotonic downtrend with mild noise — perfect for a
    short strategy to make money on."""
    import numpy as np
    rng = np.random.default_rng(123)
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(loc=-0.003, scale=0.005, size=n)))
    open_ = np.empty(n); open_[0] = 100.0; open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.001, n - 1))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = np.full(n, 1_000_000, dtype=int)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_short_strategy_profits_on_downtrend():
    """A persistent-short strategy on a synthetic downtrend should finish
    with positive total_return and a non-empty trade log."""

    @dataclass(slots=True)
    class _AlwaysShortParams:
        size: float = 1.0

    class _AlwaysShortStrategy(BaseStrategy[_AlwaysShortParams]):
        strategy_id = "_always_short_test"

        @classmethod
        def params_type(cls):
            return _AlwaysShortParams

        def indicators(self, data, params):
            return pd.DataFrame(index=data.index)

        def generate_signals(self, data, indicators, ctx: StrategyContext, params):
            df = pd.DataFrame(index=data.index)
            df["signal"] = -1
            df["signal"].iloc[0] = 0  # enter on bar 2
            df["size"] = params.size
            return SignalFrame(data=df)

    data = _downtrending_ohlcv(n=200)
    broker = Broker(ExecutionConfig(
        commission_bps=0.0, slippage_bps=0.0, allow_short=True,
    ))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    result = engine.run(_AlwaysShortStrategy(), data, _AlwaysShortParams(),
                        symbol="SYN", timeframe="1d")

    assert result.summary["total_return"] > 0, (
        f"expected positive return on downtrend short, got "
        f"{result.summary['total_return']}"
    )
    assert result.summary["n_trades"] > 0
    # Position should have been short at least once
    assert (result.positions["qty"] < 0).any()
