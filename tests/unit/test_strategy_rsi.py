from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.rsi_mean_reversion import RSIMeanReversionParams, RSIMeanReversionStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_strategy_id_and_params():
    assert RSIMeanReversionStrategy.strategy_id == "rsi_mean_reversion"
    assert RSIMeanReversionStrategy.params_type() is RSIMeanReversionParams


def test_indicator_has_rsi_column():
    data = make_ohlcv(n=200, seed=5)
    strat = RSIMeanReversionStrategy()
    p = RSIMeanReversionParams()
    ind = strat.indicators(data, p)
    assert "rsi" in ind.columns
    assert ind["rsi"].dropna().between(0, 100).all()


def test_signals_long_when_rsi_oversold():
    # Falling series -> RSI low -> oversold -> enter long after shift
    n = 100
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 - i * 0.5 for i in range(n)], "volume": 100}, index=idx)
    strat = RSIMeanReversionStrategy()
    p = RSIMeanReversionParams(period=14, oversold=30, overbought=70)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    assert sf.data["signal"].sum() > 0
