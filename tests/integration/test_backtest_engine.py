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
