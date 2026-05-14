from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtester.core.exceptions import DataError


def _yfinance_download(symbol: str, *, auto_adjust: bool, period: str, progress: bool) -> pd.DataFrame:
    """Thin indirection around yfinance.download for monkeypatching in tests.

    The yfinance package is an OPTIONAL dependency (only required for the actual
    network fetch path). The import is LOCAL so unit tests that monkeypatch
    this function don't trigger the import.
    """
    import yfinance  # local import: optional extras dependency
    return yfinance.download(
        symbol,
        period=period,
        auto_adjust=auto_adjust,
        progress=progress,
    )


def load_yfinance_cached(
    *,
    symbol: str,
    root: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
    require_volume: bool = True,
) -> pd.DataFrame:
    """Cache-on-miss yfinance loader.

    Behavior:
      - If `{root}/{symbol}.csv` exists, read it. If it covers `[start, end]`,
        slice and return. If it does NOT cover the range, raise DataError
        (explicit invalidation only — no silent re-fetch).
      - If absent, fetch via yfinance with `period="max"`, write the full
        history to `{root}/{symbol}.csv`, then slice to `[start, end]`.

    Adjustment contract: `auto_adjust=True` returns adjusted OHLC for all
    open/high/low/close columns. Volume is unadjusted.

    require_volume=False: treats zero/NaN volume as legitimate (for
    index-style symbols like ^VIX). Volume column is filled with 0 if NaN.
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    csv_path = root_p / f"{symbol}.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "timestamp"
        df = _slice_or_raise(df, symbol=symbol, start=start, end=end, csv_path=csv_path)
    else:
        raw = _yfinance_download(symbol, auto_adjust=auto_adjust, period="max", progress=False)
        df = _normalize_yfinance_frame(raw, require_volume=require_volume)
        df.to_csv(csv_path)
        df = df.loc[start:end]

    if not require_volume:
        df = df.copy()
        df["volume"] = df["volume"].fillna(0.0)

    return df


def _normalize_yfinance_frame(df: pd.DataFrame, *, require_volume: bool) -> pd.DataFrame:
    df = df.copy()
    # Drop tz if present so CSV round-trip is deterministic.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "timestamp"
    df.columns = [c.lower() for c in df.columns]
    # yfinance returns 'adj close' or 'close' depending on auto_adjust + version.
    # We keep only the canonical OHLCV; the trailing 'adj close' column (if any) is dropped.
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    if require_volume and "volume" in df.columns:
        if df["volume"].isna().any():
            raise DataError("yfinance returned NaN volume for a non-index symbol")
    return df


def _slice_or_raise(df: pd.DataFrame, *, symbol: str, start: str, end: str, csv_path: Path) -> pd.DataFrame:
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    if df.index.min() > ts_start or df.index.max() < ts_end:
        raise DataError(
            f"{symbol}.csv covers [{df.index.min().date()}, {df.index.max().date()}]; "
            f"requested [{ts_start.date()}, {ts_end.date()}]. rm the file at "
            f"{csv_path} to re-fetch."
        )
    return df.loc[start:end]
