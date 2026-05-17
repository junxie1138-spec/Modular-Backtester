from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def test_params_type_and_defaults():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    assert MLSupertrendStrategy.params_type() is MLSupertrendParams
    p = MLSupertrendParams()
    assert p.signal_mode == "reversal"
    assert p.require_new_extreme is True
    assert p.min_bars_between_signals == 10
    assert p.sensitivity == 30
    assert p.atr_period == 24
    assert p.multiplier == 1.4
    assert p.source_type == "hlcc4"
    assert p.use_atr is True
    assert p.enable_rsi is True
    assert p.rsi_len == 14
    assert p.rsi_lookback_top == 50
    assert p.rsi_lookback_bot == 50
    assert p.rsi_top == 70
    assert p.rsi_bot == 30
    assert p.vol_lookback == 3
    assert p.vol_multiplier == 1.2
    assert p.require_vol_spike is False
    assert p.enable_major_levels_only is False
    assert p.major_level_threshold == 4.5
    assert p.size == 1.0


def test_strategy_identity_and_warmup():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    s = MLSupertrendStrategy()
    assert s.strategy_id == "ml_supertrend"
    assert s.timeframe == "1d"
    # warmup = max(atr_period, sensitivity, rsi_len, vol_lookback) + 1
    assert s.warmup_bars(MLSupertrendParams()) == 31
    assert s.warmup_bars(MLSupertrendParams(atr_period=60, sensitivity=10)) == 61


def _ohlcv(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    steps = rng.normal(0.0, 1.0, n).cumsum()
    close = 100.0 + steps
    high = close + np.abs(rng.normal(0.0, 0.5, n)) + 0.5
    low = close - np.abs(rng.normal(0.0, 0.5, n)) - 0.5
    open_ = close + rng.normal(0.0, 0.3, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(800_000, 1_200_000, n).astype(float)},
        index=idx,
    )


def test_resolve_source_variants():
    from strategies.ml_supertrend import _resolve_source

    data = _ohlcv()
    assert _resolve_source(data, "close").equals(data["close"])
    assert _resolve_source(data, "high").equals(data["high"])
    hl2 = _resolve_source(data, "hl2")
    pd.testing.assert_series_equal(hl2, (data["high"] + data["low"]) / 2.0, check_names=False)
    hlcc4 = _resolve_source(data, "hlcc4")
    expected = (data["high"] + data["low"] + data["close"] + data["close"]) / 4.0
    pd.testing.assert_series_equal(hlcc4, expected, check_names=False)
    with pytest.raises(ValueError):
        _resolve_source(data, "nonsense")


def test_smoothed_tr_rma_vs_ema():
    from strategies.ml_supertrend import _smoothed_tr

    data = _ohlcv()
    rma = _smoothed_tr(data, period=14, use_atr=True)
    ema = _smoothed_tr(data, period=14, use_atr=False)
    # First bar of both equals high - low of bar 0 (TR seed).
    assert rma.iloc[0] == pytest.approx(data["high"].iloc[0] - data["low"].iloc[0])
    assert ema.iloc[0] == pytest.approx(data["high"].iloc[0] - data["low"].iloc[0])
    # No NaN (ewm with adjust=False seeds from bar 0).
    assert not rma.isna().any()
    assert not ema.isna().any()
    # RMA (alpha=1/14) and EMA (alpha=2/15) smooth differently.
    assert not np.allclose(rma.to_numpy(), ema.to_numpy())
    # All positive.
    assert (rma > 0).all() and (ema > 0).all()
