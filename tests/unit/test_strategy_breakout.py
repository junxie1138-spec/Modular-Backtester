from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.breakout_20d import Breakout20DParams, Breakout20DStrategy


def test_strategy_id_and_params():
    assert Breakout20DStrategy.strategy_id == "breakout_20d"
    assert Breakout20DStrategy.params_type() is Breakout20DParams


def test_signal_triggers_on_new_high():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    # close rises monotonically -> every bar is a new lookback high
    df = pd.DataFrame({"open": 1, "high": [100 + i for i in range(n)], "low": 1,
                       "close": [100 + i for i in range(n)], "volume": 100}, index=idx)
    strat = Breakout20DStrategy()
    p = Breakout20DParams(lookback=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # After the warmup period, signal should be 1
    assert sf.data["signal"].iloc[-1] == 1


def test_no_signal_in_warmup():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": [100] * n, "low": 1, "close": [100] * n, "volume": 100}, index=idx)
    strat = Breakout20DStrategy()
    p = Breakout20DParams(lookback=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # Flat market -> never breaks the lookback high -> no signal
    assert sf.data["signal"].sum() == 0
