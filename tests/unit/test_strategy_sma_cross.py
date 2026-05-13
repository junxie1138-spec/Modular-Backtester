from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.sma_cross import SMACrossParams, SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_strategy_id_and_params():
    assert SMACrossStrategy.strategy_id == "sma_cross"
    assert SMACrossStrategy.params_type() is SMACrossParams


def test_indicators_have_fast_and_slow():
    data = make_ohlcv(n=100, seed=4)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=10, slow=30, size=1.0)
    ind = strat.indicators(data, p)
    assert "fast_sma" in ind.columns and "slow_sma" in ind.columns


def test_signals_shifted_and_zero_when_fast_below_slow():
    # Falling series: fast SMA will be below slow SMA -> signal = 0
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 - i for i in range(n)], "volume": 100}, index=idx)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=5, slow=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # last bar should be 0 because fast < slow throughout
    assert sf.data["signal"].iloc[-1] == 0


def test_signals_one_when_fast_above_slow():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 + i for i in range(n)], "volume": 100}, index=idx)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=5, slow=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    assert sf.data["signal"].iloc[-1] == 1


def test_warmup_bars_uses_slow():
    p = SMACrossParams(fast=10, slow=50)
    assert SMACrossStrategy().warmup_bars(p) == 50
