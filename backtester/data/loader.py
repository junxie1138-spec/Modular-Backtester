from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from backtester.core.exceptions import DataError
from backtester.data.base import DataLoader


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise DataError(f"OHLCV file missing columns: {missing}")
    return df[keep]


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


class CSVDataLoader(DataLoader):
    def load(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        path = self.root / f"{symbol}.csv"
        if not path.exists():
            raise DataError(f"CSV not found for symbol {symbol!r} at {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize_ohlcv(df).sort_index()
        return _slice(df, start, end)


class ParquetDataLoader(DataLoader):
    def load(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        path = self.root / f"{symbol}.parquet"
        if not path.exists():
            raise DataError(f"Parquet not found for symbol {symbol!r} at {path}")
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = _normalize_ohlcv(df).sort_index()
        return _slice(df, start, end)


def load_symbol(
    symbol: str,
    source: str,
    root: Union[str, Path],
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    auto_adjust: bool = True,
    require_volume: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    src = source.lower()
    if src == "csv":
        return CSVDataLoader(root).load(symbol, start, end)
    if src == "parquet":
        return ParquetDataLoader(root).load(symbol, start, end)
    if src == "yfinance":
        from backtester.data.yfinance_loader import load_yfinance_cached
        if start is None or end is None:
            raise DataError("yfinance loader requires explicit start and end dates")
        return load_yfinance_cached(
            symbol=symbol, root=str(root), start=start, end=end,
            auto_adjust=auto_adjust, require_volume=require_volume,
        )
    raise DataError(f"unknown source: {source!r} (allowed: csv, parquet, yfinance)")
