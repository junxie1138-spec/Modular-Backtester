from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.hourly_stitch import SeamReport, load_donor, validate_seam

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


def test_validate_seam_constant_ratio_passes() -> None:
    idx = _session_index("2024-01-02", 40)
    close = 100.0 + np.arange(len(idx)) * 0.01
    recent = _ohlcv(idx, close)
    donor = _ohlcv(idx, close * 2.0)   # a constant 2x adjustment offset
    rep = validate_seam(donor, recent)
    assert rep.ok
    assert rep.scale == pytest.approx(2.0)
    assert rep.offset_hours == 0


def test_validate_seam_no_overlap_aborts() -> None:
    donor = _ohlcv(_session_index("2024-01-02", 20), np.full(20 * 7, 100.0))
    recent = _ohlcv(_session_index("2024-06-03", 20), np.full(20 * 7, 100.0))
    rep = validate_seam(donor, recent)
    assert not rep.ok and "no overlap" in rep.reason


def test_validate_seam_extended_hours_aborts() -> None:
    ext = [float(h) for h in range(7, 20)]   # 13 bars/day
    donor = _ohlcv(_session_index("2024-01-02", 30, hours=ext),
                   np.full(30 * 13, 100.0))
    recent = _ohlcv(_session_index("2024-01-02", 30), np.full(30 * 7, 100.0))
    rep = validate_seam(donor, recent)
    assert not rep.ok and "bars/day" in rep.reason


def test_validate_seam_close_stamped_donor_detected_and_passes() -> None:
    recent_idx = _session_index("2024-01-02", 40)
    close = 100.0 + np.arange(len(recent_idx)) * 0.01
    recent = _ohlcv(recent_idx, close)
    # A close-stamped donor: every bar is stamped one hour later.
    donor = _ohlcv(recent_idx + pd.Timedelta(hours=1), close)
    rep = validate_seam(donor, recent)
    assert rep.ok
    assert rep.offset_hours == -1


def test_validate_seam_irregular_timestamps_abort() -> None:
    bad = [h + 0.25 for h in _HOURS]   # 09:45, 10:45, ... — off the grid
    donor = _ohlcv(_session_index("2024-01-02", 40, hours=bad),
                   np.full(40 * 7, 100.0))
    recent = _ohlcv(_session_index("2024-01-02", 40), np.full(40 * 7, 100.0))
    rep = validate_seam(donor, recent)
    assert not rep.ok and "09:30 grid" in rep.reason


def test_validate_seam_single_outlier_tolerated() -> None:
    idx = _session_index("2024-01-02", 40)
    recent = _ohlcv(idx, np.full(len(idx), 100.0))
    donor_close = np.full(len(idx), 100.0)
    donor_close[5] = 150.0   # one bad bar
    rep = validate_seam(_ohlcv(idx, donor_close), recent)
    assert rep.ok            # robust median + MAD ignore the lone outlier


def test_validate_seam_drifting_ratio_aborts() -> None:
    idx = _session_index("2024-01-02", 40)
    n = len(idx)
    recent = _ohlcv(idx, np.full(n, 100.0))
    donor = _ohlcv(idx, 100.0 * np.linspace(0.9, 1.1, n))   # ratio drifts
    rep = validate_seam(donor, recent)
    assert not rep.ok and "scale dispersion" in rep.reason


def test_validate_seam_tail_disagreement_aborts_on_agreement() -> None:
    idx = _session_index("2024-01-02", 40)
    n = len(idx)
    recent = _ohlcv(idx, np.full(n, 100.0))
    donor_close = np.full(n, 100.0)
    donor_close[: n // 8] = 102.0   # ~12% of bars off by 2% (MAD stays ~0)
    rep = validate_seam(_ohlcv(idx, donor_close), recent)
    assert not rep.ok and "agreement" in rep.reason


def test_validate_seam_empty_frame_aborts() -> None:
    idx = _session_index("2024-01-02", 10)
    recent = _ohlcv(idx, np.full(len(idx), 100.0))
    empty = recent.iloc[:0]
    assert not validate_seam(empty, recent).ok
    assert not validate_seam(recent, empty).ok
    assert "empty" in validate_seam(empty, recent).reason


def test_validate_seam_pre_open_stamped_donor_detected_and_passes() -> None:
    recent_idx = _session_index("2024-01-02", 40)
    close = 100.0 + np.arange(len(recent_idx)) * 0.01
    recent = _ohlcv(recent_idx, close)
    # A pre-open-stamped donor: every bar stamped one hour earlier than the
    # 09:30 grid -> validate_seam corrects it with a +1h offset.
    donor = _ohlcv(recent_idx - pd.Timedelta(hours=1), close)
    rep = validate_seam(donor, recent)
    assert rep.ok
    assert rep.offset_hours == 1
