from __future__ import annotations

import pandas as pd

from backtester.core.constants import REQUIRED_OHLCV_COLUMNS
from backtester.core.exceptions import DataError


def validate_ohlcv(df: pd.DataFrame, *, strict_volume: bool = True) -> None:
    cols = set(map(str.lower, df.columns))
    missing = set(REQUIRED_OHLCV_COLUMNS) - cols
    if missing:
        raise DataError(f"data missing columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("data index must be a DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise DataError("data index must be monotonic increasing")

    if df.index.duplicated().any():
        raise DataError("data index contains duplicate timestamps")

    price_cols = ["open", "high", "low", "close"]
    if df[price_cols].isna().any().any():
        raise DataError("data contains NaN values")

    if (df[price_cols] <= 0).any().any():
        raise DataError("data contains non-positive prices")

    if strict_volume:
        if df["volume"].isna().any():
            raise DataError("data contains NaN values")
        if (df["volume"] < 0).any():
            raise DataError("validate_ohlcv: volume contains negative values")
    else:
        # When strict_volume=False, volume may be zero/NaN; coerce NaN to 0
        # to keep downstream serialization clean.
        df["volume"] = df["volume"].fillna(0.0)
