"""Pure, network-free helpers for stitching a deep-history donor hourly CSV
with a recent yfinance 1h feed.

Three functions: `load_donor` normalizes a Kaggle donor CSV to the OHLCV
contract; `validate_seam` runs five robustness checks on the donor/yfinance
overlap; `splice` cuts the donor before the seam and takes yfinance from the
seam onward. None of them touch the network or write files — the build script
(scripts/build_hourly_dataset.py) owns that.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.core.exceptions import DataError

# yfinance 1h regular session: 7 bars/day, open-stamped 09:30..15:30 Eastern.
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(15, 30)
# A donor averaging more bars/day than this still carries extended hours.
_MAX_BARS_PER_DAY = 9
# Below this many overlapping bars the robust statistics are meaningless.
MIN_OVERLAP_BARS = 7
# Seam-validation tolerances. Tunable; see the hourly-timeframe design spec.
SCALE_DISPERSION_TOL = 0.01   # MAD of the donor/yfinance ratio about its median
AGREEMENT_TOL = 0.005         # 95th-pct post-scale absolute relative close error
# An hourly stitch must not introduce a calendar gap larger than this.
MAX_SEAM_GAP = pd.Timedelta(days=5)

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class SeamReport:
    """The verdict of validate_seam. Makes no tradability claim — only reports
    whether the donor and yfinance frames can be safely joined."""
    ok: bool
    scale: float
    offset_hours: int
    overlap_bars: int
    scale_dispersion: float
    agreement_error: float
    reason: str


def load_donor(path: str | Path) -> pd.DataFrame:
    """Read a Kaggle donor CSV and normalize it to the OHLCV contract.

    Returns a frame with lowercase open/high/low/close/volume columns and a
    tz-naive US/Eastern DatetimeIndex filtered to the regular session
    (09:30-15:30, inclusive), sorted ascending, de-duplicated. A tz-aware
    donor index is converted to US/Eastern before the tz label is dropped; a
    tz-naive index is assumed to already be US/Eastern (a human eyeballs each
    donor before placing it — see the design spec's provenance note).
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"donor CSV not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError(f"donor CSV {path.name}: index is not datetime-parseable")
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in _OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"donor CSV {path.name} missing columns: {missing}")
    df = df[_OHLCV_COLUMNS].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    times = df.index.time
    in_session = (times >= _SESSION_OPEN) & (times <= _SESSION_CLOSE)
    return df[in_session]
