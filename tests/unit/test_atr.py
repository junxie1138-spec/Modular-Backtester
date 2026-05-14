from __future__ import annotations

import pandas as pd
import pytest


def _ohlc(highs, lows, closes):
    return pd.DataFrame({
        "open": closes,  # not used by ATR
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * len(closes),
    })


def test_atr_first_value_is_high_minus_low():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0, 11.0], lows=[8.0, 9.0], closes=[9.0, 10.0])
    atr = compute_atr(data, period=2)
    # TR[0] = 10 - 8 = 2 (by convention; close[-1] undefined)
    # TR[1] = max(11-9, |11-9|, |9-9|) = 2
    # ATR with period=2 first defined at index 1: mean(TR[0:2]) = 2.0
    assert atr.iloc[1] == pytest.approx(2.0)


def test_atr_period_2_matches_hand_calc():
    from backtester.engine.atr import compute_atr
    data = _ohlc(
        highs=[10.0, 12.0, 14.0, 11.0, 13.0],
        lows=[8.0, 10.0, 11.0, 9.0, 11.0],
        closes=[9.0, 11.0, 13.0, 10.0, 12.0],
    )
    # TR[0] = 10 - 8 = 2
    # TR[1] = max(12-10, |12-9|, |10-9|) = max(2,3,1) = 3
    # TR[2] = max(14-11, |14-11|, |11-11|) = max(3,3,0) = 3
    # TR[3] = max(11-9, |11-13|, |9-13|) = max(2,2,4) = 4
    # TR[4] = max(13-11, |13-10|, |11-10|) = max(2,3,1) = 3
    atr = compute_atr(data, period=2)
    assert atr.iloc[1] == pytest.approx((2 + 3) / 2)
    assert atr.iloc[2] == pytest.approx((3 + 3) / 2)
    assert atr.iloc[3] == pytest.approx((3 + 4) / 2)
    assert atr.iloc[4] == pytest.approx((4 + 3) / 2)


def test_atr_nan_before_period():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0, 11.0, 12.0], lows=[8.0, 9.0, 10.0], closes=[9.0, 10.0, 11.0])
    atr = compute_atr(data, period=3)
    assert pd.isna(atr.iloc[0])
    assert pd.isna(atr.iloc[1])
    assert not pd.isna(atr.iloc[2])


def test_atr_invalid_period_raises():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0], lows=[8.0], closes=[9.0])
    with pytest.raises(ValueError, match="ATR period must be >= 2"):
        compute_atr(data, period=1)
