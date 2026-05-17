from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pandas as pd
import pytest


def _fake_yf_history(symbol: str, **_: object) -> pd.DataFrame:
    """yfinance-style frame: tz-aware index, columns Open/High/Low/Close/Volume."""
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low":  [ 99.0, 100.0, 101.0, 102.0, 103.0],
            "Close":[100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume":[1_000_000] * 5,
        },
        index=idx,
    )


def test_loader_reads_cache_when_csv_present(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached

    # Pre-populate the cache.
    sym_path = tmp_path / "FAKE.csv"
    fake_csv = _fake_yf_history("FAKE").rename(
        columns={c: c.lower() for c in ["Open", "High", "Low", "Close", "Volume"]}
    )
    fake_csv.index.name = "timestamp"
    # Drop tz so CSV round-trip is clean.
    fake_csv.index = fake_csv.index.tz_localize(None)
    fake_csv.to_csv(sym_path)

    with patch("backtester.data.yfinance_loader._yfinance_download") as mock_dl:
        df = load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-02", end="2020-01-08",
        )
        mock_dl.assert_not_called()  # cache hit — no network
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_loader_fetches_via_yfinance_when_csv_absent(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_fake_yf_history) as mock_dl:
        df = load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
        )
        mock_dl.assert_called_once()
    assert (tmp_path / "FAKE.csv").exists()
    assert len(df) == 5


def test_loader_raises_when_cache_does_not_cover_range(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    sym_path = tmp_path / "FAKE.csv"
    fake_csv = _fake_yf_history("FAKE").rename(
        columns={c: c.lower() for c in ["Open", "High", "Low", "Close", "Volume"]}
    )
    fake_csv.index.name = "timestamp"
    fake_csv.index = fake_csv.index.tz_localize(None)
    fake_csv.to_csv(sym_path)

    with pytest.raises(Exception, match="rm the file"):
        load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2019-01-01", end="2019-12-31",
        )


def test_loader_auto_adjust_true_passes_through(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_fake_yf_history) as mock_dl:
        load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
            auto_adjust=True,
        )
        kwargs = mock_dl.call_args.kwargs
        assert kwargs.get("auto_adjust") is True


def test_loader_require_volume_false_keeps_zero_volume(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    def _vix_history(symbol, **_):
        df = _fake_yf_history(symbol)
        df["Volume"] = 0
        return df
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_vix_history):
        df = load_yfinance_cached(
            symbol="^VIX", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
            require_volume=False,
        )
    assert (df["volume"] == 0).all()


def test_yfinance_download_forwards_interval_and_prepost(monkeypatch) -> None:
    import backtester.data.yfinance_loader as yl
    captured: dict = {}

    class _FakeYF:
        @staticmethod
        def download(symbol, **kwargs):
            captured.update(symbol=symbol, **kwargs)
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF)
    yl._yfinance_download(
        "SPY", auto_adjust=True, period="730d", progress=False,
        interval="1h", prepost=False,
    )
    assert captured["interval"] == "1h"
    assert captured["prepost"] is False
    assert captured["period"] == "730d"


def test_yfinance_download_defaults_preserve_daily(monkeypatch) -> None:
    """Omitting interval/prepost keeps the existing daily fetch behaviour."""
    import backtester.data.yfinance_loader as yl
    captured: dict = {}

    class _FakeYF:
        @staticmethod
        def download(symbol, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF)
    yl._yfinance_download("SPY", auto_adjust=True, period="max", progress=False)
    assert captured["interval"] == "1d"
    assert captured["prepost"] is False
