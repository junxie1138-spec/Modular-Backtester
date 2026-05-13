from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.loader import CSVDataLoader, ParquetDataLoader, load_symbol
from tests.fixtures.synthetic import make_ohlcv


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    df = make_ohlcv(n=120, seed=2)
    df.to_csv(tmp_path / "SPY.csv", index_label="date")
    return tmp_path


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    df = make_ohlcv(n=120, seed=2)
    df.to_parquet(tmp_path / "SPY.parquet", index=True)
    return tmp_path


def test_csv_loader_loads_symbol(csv_dir):
    loader = CSVDataLoader(root=csv_dir)
    df = loader.load("SPY")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 120


def test_csv_loader_respects_date_range(csv_dir):
    loader = CSVDataLoader(root=csv_dir)
    raw = loader.load("SPY")
    mid_start = raw.index[30].strftime("%Y-%m-%d")
    mid_end = raw.index[60].strftime("%Y-%m-%d")
    df = loader.load("SPY", start=mid_start, end=mid_end)
    assert df.index.min() >= pd.Timestamp(mid_start)
    assert df.index.max() <= pd.Timestamp(mid_end)


def test_csv_loader_missing_symbol_raises(tmp_path):
    loader = CSVDataLoader(root=tmp_path)
    with pytest.raises(DataError, match="MISSING"):
        loader.load("MISSING")


def test_parquet_loader_loads_symbol(parquet_dir):
    loader = ParquetDataLoader(root=parquet_dir)
    df = loader.load("SPY")
    assert len(df) == 120
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_load_symbol_dispatches_by_source(csv_dir):
    df = load_symbol(symbol="SPY", source="csv", root=csv_dir)
    assert len(df) == 120


def test_load_symbol_unknown_source(tmp_path):
    with pytest.raises(DataError, match="unknown source"):
        load_symbol(symbol="SPY", source="hdf5", root=tmp_path)
