from __future__ import annotations

import pandas as pd


def compute_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using simple-moving-average smoothing.

    TR[i] = max(high - low, |high - prev_close|, |low - prev_close|).
    TR[0] uses high[0] - low[0] (prev_close undefined).
    ATR[i] = rolling SMA of TR over `period` bars.

    Returns a Series aligned to data.index. First `period - 1` values
    are NaN. Callers MUST treat NaN as "ATR not yet available".
    """
    if period < 2:
        raise ValueError("ATR period must be >= 2")
    high = data["high"]
    low = data["low"]
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.rolling(period).mean()
