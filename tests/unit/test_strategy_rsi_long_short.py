from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def _make_synthetic_oscillating():
    """50 bars of price that swings up then down then up again so RSI
    crosses both thresholds."""
    n = 80
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    # V-shape: down 30 days, up 30 days, down 20 days
    prices = []
    p = 100.0
    for i in range(n):
        if i < 30:
            p *= 0.985  # strong down
        elif i < 60:
            p *= 1.015  # strong up
        else:
            p *= 0.985
        prices.append(p)
    df = pd.DataFrame({
        "open": prices, "high": [pr * 1.01 for pr in prices],
        "low": [pr * 0.99 for pr in prices], "close": prices,
        "volume": [1_000_000] * n,
    }, index=idx)
    return df


def test_rsi_long_short_emits_long_and_short_signals():
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )

    data = _make_synthetic_oscillating()
    strat = RSILongShortStrategy()
    params = RSILongShortParams(period=7, oversold=30.0, overbought=70.0)
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d", warmup_bars=strat.warmup_bars(params))
    sf = strat.generate_signals(data, ind, ctx, params)

    sigs = sf.data["signal"]
    assert (sigs == 1).any(), "expected at least one long signal"
    assert (sigs == -1).any(), "expected at least one short signal"
    assert set(sigs.unique()).issubset({-1, 0, 1})


def test_rsi_long_short_signal_is_shifted_by_one_bar():
    """Following the same convention as the existing strategies:
    signal at bar i corresponds to a decision based on bar i-1's data."""
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )

    data = _make_synthetic_oscillating()
    strat = RSILongShortStrategy()
    params = RSILongShortParams(period=7)
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d", warmup_bars=strat.warmup_bars(params))
    sf = strat.generate_signals(data, ind, ctx, params)
    # First bar is always flat after a shift
    assert sf.data["signal"].iloc[0] == 0


def test_rsi_long_short_warmup_bars_matches_period():
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )
    s = RSILongShortStrategy()
    assert s.warmup_bars(RSILongShortParams(period=14)) == 15
