from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scripts.build_hourly_dataset as bh

_HOURS = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]


def _session_index(start: str, n_days: int, hours: list[float] = _HOURS) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=n_days)
    return pd.DatetimeIndex(
        [d + pd.Timedelta(hours=h) for d in days for h in hours]
    )


def _ohlcv(index: pd.DatetimeIndex, close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(len(close), 1000.0),
    }, index=index)


def test_build_symbol_yfinance_only_thin_is_insufficient(tmp_path, monkeypatch) -> None:
    recent = _ohlcv(_session_index("2024-01-02", 200), np.full(200 * 7, 100.0))
    monkeypatch.setattr(bh, "_yfinance_download", lambda *a, **k: recent.copy())
    entry = bh.build_symbol(
        "SPY", donor_dir=tmp_path / "no_donor", out_dir=tmp_path / "out",
    )
    assert entry["source"] == "yfinance_only"
    assert entry["classification"] == "insufficient_history"
    assert (tmp_path / "out" / "SPY.csv").exists()


def test_build_symbol_stitches_when_donor_present(tmp_path, monkeypatch) -> None:
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    out_dir = tmp_path / "out"
    recent = _ohlcv(_session_index("2023-01-02", 360), np.full(360 * 7, 100.0))
    monkeypatch.setattr(bh, "_yfinance_download", lambda *a, **k: recent.copy())
    # A deep donor overlapping the recent window with matching prices.
    donor_idx = _session_index("2016-01-04", 1900)
    donor = _ohlcv(donor_idx, np.full(len(donor_idx), 100.0))
    donor.to_csv(donor_dir / "SPY.csv", index_label="timestamp")

    entry = bh.build_symbol("SPY", donor_dir=donor_dir, out_dir=out_dir)
    assert entry["source"] == "stitched"
    assert entry["classification"] == "tradable"
    assert entry["bar_count"] >= bh.MIN_HOURLY_BARS
    assert (out_dir / "SPY.csv").exists()


def test_build_symbol_falls_back_when_seam_rejected(tmp_path, monkeypatch) -> None:
    donor_dir = tmp_path / "donor"
    donor_dir.mkdir()
    recent = _ohlcv(_session_index("2023-01-02", 360), np.full(360 * 7, 100.0))
    monkeypatch.setattr(bh, "_yfinance_download", lambda *a, **k: recent.copy())
    # An oscillating donor: the donor/yfinance ratio swings wildly -> the
    # seam's scale-dispersion check rejects it.
    donor_idx = _session_index("2016-01-04", 1900)
    donor = _ohlcv(donor_idx, 100.0 + 30.0 * np.sin(np.arange(len(donor_idx)) / 5.0))
    donor.to_csv(donor_dir / "SPY.csv", index_label="timestamp")

    entry = bh.build_symbol("SPY", donor_dir=donor_dir, out_dir=tmp_path / "out")
    assert entry["source"] == "yfinance_only"
    assert "seam rejected" in entry["validation"]
    assert (tmp_path / "out" / "SPY.csv").exists()


def test_main_writes_build_report(tmp_path, monkeypatch) -> None:
    recent = _ohlcv(_session_index("2024-01-02", 200), np.full(200 * 7, 100.0))
    monkeypatch.setattr(bh, "_yfinance_download", lambda *a, **k: recent.copy())
    out_dir = tmp_path / "raw_hourly"
    rc = bh.main([
        "--symbols", "SPY", "QQQ",
        "--donor-dir", str(tmp_path / "donor"),
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    report = json.loads((out_dir / "_build_report.json").read_text(encoding="utf-8"))
    assert set(report["symbols"]) == {"SPY", "QQQ"}
    assert report["min_hourly_bars"] == bh.MIN_HOURLY_BARS
    assert report["symbols"]["SPY"]["source"] == "yfinance_only"
    assert report["symbols"]["SPY"]["classification"] == "insufficient_history"
