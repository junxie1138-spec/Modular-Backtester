from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.hourly_stitch import load_donor

_HOURS = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]


def _session_index(start: str, n_days: int, hours: list[float] = _HOURS) -> pd.DatetimeIndex:
    """Regular-session-style hourly timestamps: n_days business days x `hours`."""
    days = pd.bdate_range(start, periods=n_days)
    return pd.DatetimeIndex(
        [d + pd.Timedelta(hours=h) for d in days for h in hours]
    )


def _ohlcv(index: pd.DatetimeIndex, close) -> pd.DataFrame:
    """Build a contract-valid OHLCV frame from a close-price array."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(len(close), 1000.0),
    }, index=index)


def test_load_donor_normalizes_columns_and_filters_session(tmp_path) -> None:
    # Donor CSV with extended-hours rows (08:00, 17:00) and mixed-case columns.
    idx = _session_index("2024-01-02", 5, hours=[8.0] + _HOURS + [17.0])
    raw = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
        "Volume": 1000.0,
    }, index=idx)
    path = tmp_path / "SPY.csv"
    raw.to_csv(path, index_label="timestamp")

    df = load_donor(path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # 08:00 and 17:00 rows dropped -> 7 regular-session bars per day remain.
    assert len(df) == 5 * 7
    times = set(pd.Series(df.index.time).unique())
    assert times == {pd.Timestamp(f"2024-01-02 {h:02d}:30").time()
                     for h in range(9, 16)}


def test_load_donor_missing_file_raises(tmp_path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_donor(tmp_path / "nope.csv")


def test_load_donor_missing_columns_raise(tmp_path) -> None:
    path = tmp_path / "BAD.csv"
    pd.DataFrame(
        {"open": [1.0], "close": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02 09:30"]),
    ).to_csv(path, index_label="timestamp")
    with pytest.raises(DataError, match="missing columns"):
        load_donor(path)
