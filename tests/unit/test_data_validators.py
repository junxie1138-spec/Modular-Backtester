from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.validators import validate_ohlcv
from tests.fixtures.synthetic import make_ohlcv


def test_validate_passes_on_clean_data(ohlcv_small):
    validate_ohlcv(ohlcv_small)


def test_validate_rejects_missing_columns(ohlcv_small):
    bad = ohlcv_small.drop(columns=["close"])
    with pytest.raises(DataError, match="missing columns"):
        validate_ohlcv(bad)


def test_validate_rejects_non_datetime_index(ohlcv_small):
    bad = ohlcv_small.reset_index(drop=True)
    with pytest.raises(DataError, match="DatetimeIndex"):
        validate_ohlcv(bad)


def test_validate_rejects_non_monotonic_index(ohlcv_small):
    bad = ohlcv_small.iloc[::-1]
    with pytest.raises(DataError, match="monotonic"):
        validate_ohlcv(bad)


def test_validate_rejects_duplicates(ohlcv_small):
    bad = pd.concat([ohlcv_small, ohlcv_small.head(1)]).sort_index()
    with pytest.raises(DataError, match="duplicate"):
        validate_ohlcv(bad)


def test_validate_rejects_negative_prices(ohlcv_small):
    bad = ohlcv_small.copy()
    bad.iloc[5, bad.columns.get_loc("low")] = -1.0
    with pytest.raises(DataError, match="non-positive"):
        validate_ohlcv(bad)


def test_validate_rejects_nan(ohlcv_small):
    bad = ohlcv_small.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(DataError, match="NaN"):
        validate_ohlcv(bad)


def _make_index_style_frame(rows: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [0.0] * rows,  # index-style: no volume
        },
        index=idx,
    )


def test_validator_default_rejects_negative_volume():
    df = _make_index_style_frame()
    df.iloc[5, df.columns.get_loc("volume")] = -1.0  # negative volume should fail
    with pytest.raises(DataError):
        validate_ohlcv(df)


def test_validator_strict_volume_false_allows_zero_volume():
    df = _make_index_style_frame()
    validate_ohlcv(df, strict_volume=False)  # must not raise with zero volume
