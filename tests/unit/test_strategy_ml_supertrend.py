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


def test_supertrend_trend_range_and_start():
    from strategies.ml_supertrend import _supertrend_trend

    n = 60
    close = np.full(n, 100.0)
    src = close.copy()
    atr = np.full(n, 2.0)
    trend = _supertrend_trend(src, atr, close, multiplier=1.0)
    assert trend.shape == (n,)
    assert set(np.unique(trend)).issubset({-1, 1})
    # On a flat series price never crosses a band, trend stays at its +1 start.
    assert trend[0] == 1
    assert (trend == 1).all()


def test_supertrend_trend_flips_down_then_up():
    from strategies.ml_supertrend import _supertrend_trend

    # Rise for 30 bars, then a sharp sustained fall, then a sharp rise.
    up = np.linspace(100.0, 130.0, 30)
    down = np.linspace(130.0, 70.0, 30)
    up2 = np.linspace(70.0, 110.0, 30)
    close = np.concatenate([up, down, up2])
    src = close.copy()
    atr = np.full(close.shape[0], 2.0)
    trend = _supertrend_trend(src, atr, close, multiplier=1.0)
    assert trend[0] == 1
    # Somewhere in the decline the trend must flip to -1.
    assert (trend[30:60] == -1).any(), "expected a downtrend during the fall"
    # And flip back to +1 during the final rise.
    assert (trend[60:] == 1).any(), "expected an uptrend during the recovery"


def test_wilder_rsi_range_and_warmup():
    from strategies.ml_supertrend import _wilder_rsi

    data = _ohlcv(n=60, seed=3)
    rsi = _wilder_rsi(data["close"], length=14)
    valid = rsi.dropna()
    # First `length` values are NaN (min_periods=length).
    assert rsi.iloc[:13].isna().all()
    assert not rsi.iloc[14:].isna().any()
    # RSI is bounded to [0, 100].
    assert (valid >= 0).all() and (valid <= 100).all()


def test_wilder_rsi_all_gains_is_100():
    from strategies.ml_supertrend import _wilder_rsi

    # Monotonically rising close -> no losses -> RSI saturates at 100.
    close = pd.Series(np.linspace(100.0, 160.0, 40))
    rsi = _wilder_rsi(close, length=14)
    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)


_IND_COLUMNS = {
    "atr", "st_trend", "rsi", "roll_high", "roll_low",
    "is_new_high", "is_new_low", "rsi_cold", "rsi_hot",
    "vol_surge", "sig_high", "sig_low",
}


def test_indicators_produces_all_columns():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=120, seed=5)
    ind = MLSupertrendStrategy().indicators(data, MLSupertrendParams())
    assert _IND_COLUMNS.issubset(set(ind.columns))
    assert len(ind) == len(data)
    assert set(np.unique(ind["st_trend"].to_numpy())).issubset({-1, 1})
    assert ind["atr"].dropna().gt(0).all()
    for col in ("is_new_high", "is_new_low", "rsi_cold", "rsi_hot",
                "vol_surge", "sig_high", "sig_low"):
        assert ind[col].dtype == bool


def test_indicators_rsi_filters_off_when_disabled():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=120, seed=6)
    ind = MLSupertrendStrategy().indicators(
        data, MLSupertrendParams(enable_rsi=False)
    )
    # With RSI disabled both filter columns are constant True.
    assert ind["rsi_cold"].all()
    assert ind["rsi_hot"].all()


def test_indicators_major_levels_filter_is_subset():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=200, seed=7)
    strat = MLSupertrendStrategy()
    base = strat.indicators(data, MLSupertrendParams(enable_major_levels_only=False))
    major = strat.indicators(data, MLSupertrendParams(enable_major_levels_only=True))
    # The key-levels filter can only remove fresh extremes, never add them.
    assert (major["sig_high"] <= base["sig_high"]).all()
    assert (major["sig_low"] <= base["sig_low"]).all()
    # When the filter is off, sig_* equals is_new_* exactly.
    assert base["sig_high"].equals(base["is_new_high"])
    assert base["sig_low"].equals(base["is_new_low"])


def _swinging_ohlcv(n=300, seed=11):
    """Trend-swinging series: repeated up/down legs so SuperTrend flips
    multiple times and fresh highs/lows occur in both directions."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.empty(n)
    p = 100.0
    for i in range(n):
        leg = (i // 25) % 2          # alternate 25-bar up / down legs
        drift = 0.6 if leg == 0 else -0.6
        p = max(5.0, p + drift + rng.normal(0.0, 0.4))
        close[i] = p
    high = close + np.abs(rng.normal(0.0, 0.4, n)) + 0.3
    low = close - np.abs(rng.normal(0.0, 0.4, n)) - 0.3
    open_ = close + rng.normal(0.0, 0.2, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(800_000, 1_200_000, n).astype(float)},
        index=idx,
    )


def _run(strat, data, params):
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d",
                          warmup_bars=strat.warmup_bars(params))
    return strat.generate_signals(data, ind, ctx, params).data["signal"]


def test_reversal_signal_values_and_first_bar_flat():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data, MLSupertrendParams(signal_mode="reversal"))
    assert set(sigs.unique()).issubset({-1, 0, 1})
    assert sigs.iloc[0] == 0          # shift(1) leaves the first bar flat
    assert len(sigs) == len(data)


def test_reversal_no_signal_in_warmup():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    strat = MLSupertrendStrategy()
    params = MLSupertrendParams(signal_mode="reversal")
    sigs = _run(strat, data, params)
    # Every bar up to and including warmup index is flat.
    assert (sigs.iloc[: strat.warmup_bars(params) + 1] == 0).all()


def test_reversal_is_stop_and_reverse():
    """Once trading starts the position is never flat again, and consecutive
    distinct non-zero values alternate +1 / -1 (stop-and-reverse)."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data,
                MLSupertrendParams(signal_mode="reversal")).to_numpy()
    nz = sigs[sigs != 0]
    assert nz.size > 0, "expected at least one signal on a swinging series"
    # Collapse runs of equal values; the distinct sequence must alternate.
    collapsed = nz[np.insert(np.diff(nz) != 0, 0, True)]
    assert np.all(np.abs(np.diff(collapsed)) == 2), "non-zero signal must alternate +1/-1"
    # After the first signal there is no return to flat.
    first = np.argmax(sigs != 0)
    assert np.all(sigs[first:] != 0)


def test_reversal_signal_spacing_is_honoured():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    params = MLSupertrendParams(signal_mode="reversal", min_bars_between_signals=20)
    sigs = _run(MLSupertrendStrategy(), data, params).to_numpy()
    # A "new signal bar" is where the held position changes value.
    change_idx = np.where(np.diff(sigs) != 0)[0] + 1
    # Drop the initial 0 -> first-signal transition is still a real signal;
    # gaps between successive signal bars must be >= min_bars_between_signals.
    gaps = np.diff(change_idx)
    assert np.all(gaps >= params.min_bars_between_signals)


def test_breakout_emits_signals_and_is_stop_and_reverse():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data,
                MLSupertrendParams(signal_mode="breakout")).to_numpy()
    assert set(np.unique(sigs)).issubset({-1, 0, 1})
    nz = sigs[sigs != 0]
    assert nz.size > 0, "expected at least one breakout signal"
    collapsed = nz[np.insert(np.diff(nz) != 0, 0, True)]
    assert np.all(np.abs(np.diff(collapsed)) == 2), "breakout signal must alternate +1/-1"


def test_reversal_and_breakout_differ_on_same_series():
    """Spec section 8.1: on the same series the two modes fire at different
    times (Breakout fires on the fresh-extreme bar, Reversal later on the
    confirmed trend flip), so the held-position series are not identical."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    strat = MLSupertrendStrategy()
    rev = _run(strat, data, MLSupertrendParams(signal_mode="reversal"))
    brk = _run(strat, data, MLSupertrendParams(signal_mode="breakout"))
    assert not rev.equals(brk), "reversal and breakout must not produce identical signals"


def test_require_vol_spike_blocks_signals_without_surges():
    """With constant volume there is never a surge, so require_vol_spike=True
    blocks every signal. The same series without the gate still trades."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv().copy()
    data["volume"] = 1_000_000.0          # constant -> vol_surge is always False
    strat = MLSupertrendStrategy()

    gated = _run(strat, data,
                 MLSupertrendParams(signal_mode="breakout", require_vol_spike=True))
    assert (gated == 0).all(), "no surge -> require_vol_spike blocks every signal"

    ungated = _run(strat, data,
                   MLSupertrendParams(signal_mode="breakout", require_vol_spike=False))
    assert (ungated != 0).any(), "without the vol gate the series still trades"


def test_ml_supertrend_is_registered():
    from backtester.strategies.registry import get_strategy_class
    from strategies.ml_supertrend import MLSupertrendStrategy

    assert get_strategy_class("ml_supertrend") is MLSupertrendStrategy
