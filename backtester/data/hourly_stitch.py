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


def validate_seam(donor: pd.DataFrame, recent: pd.DataFrame) -> SeamReport:
    """Run the five seam checks on the donor/recent overlap window.

    `donor` is a load_donor() output; `recent` is a normalized yfinance 1h
    frame (tz-naive US/Eastern, regular session, open-stamped). The five
    checks run in order: (1) overlap exists, (2) donor is regular-session,
    (3) donor is open-stamped on the 09:30 grid, (4) the donor/yfinance close
    ratio has a trustworthy median (MAD dispersion), (5) post-scale closes
    agree (95th-pct relative error). Returns a SeamReport; makes no
    tradability claim.
    """
    def _fail(reason: str, overlap_bars: int = 0) -> SeamReport:
        return SeamReport(
            ok=False, scale=1.0, offset_hours=0, overlap_bars=overlap_bars,
            scale_dispersion=float("nan"), agreement_error=float("nan"),
            reason=reason,
        )

    if len(donor) == 0 or len(recent) == 0:
        return _fail("donor or recent frame is empty")

    # Check 1: the donor must not end before yfinance's coverage begins.
    if donor.index.max() < recent.index.min():
        return _fail("no overlap: donor ends before yfinance coverage begins")

    # Check 2: regular session — the donor must average ~7 bars/day.
    n_days = max(1, donor.index.normalize().nunique())
    bars_per_day = len(donor) / n_days
    if bars_per_day > _MAX_BARS_PER_DAY:
        return _fail(
            f"donor averages {bars_per_day:.1f} bars/day — extended hours "
            "not cleanly filtered"
        )

    # Check 3: timestamp convention — donor open-stamped on the 09:30 grid.
    # A close-stamped donor is offset a uniform +1h; correct it with -1h.
    recent_times = set(recent.index.time)
    offset_hours: int | None = None
    for cand in (0, -1, 1):
        shifted = set((donor.index + pd.Timedelta(hours=cand)).time)
        if shifted <= recent_times:
            offset_hours = cand
            break
    if offset_hours is None:
        return _fail("donor intraday timestamps do not align to the 09:30 grid")

    aligned = donor.copy()
    if offset_hours != 0:
        aligned.index = aligned.index + pd.Timedelta(hours=offset_hours)

    overlap = aligned.index.intersection(recent.index)
    if len(overlap) < MIN_OVERLAP_BARS:
        return _fail(
            f"only {len(overlap)} overlapping bars (need >= {MIN_OVERLAP_BARS})",
            len(overlap),
        )

    d_close = aligned.loc[overlap, "close"].astype(float)
    r_close = recent.loc[overlap, "close"].astype(float)
    ratio = (d_close / r_close).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < MIN_OVERLAP_BARS:
        return _fail("overlap closes are not comparable (zero/NaN closes)", len(overlap))
    scale = float(ratio.median())
    if not np.isfinite(scale) or scale <= 0:
        return _fail(f"robust scale is non-finite or non-positive ({scale})", len(overlap))

    # Check 4: robust dispersion — MAD of the ratio about its median. A
    # constant offset is fine (scale absorbs it); a drift is not.
    scale_dispersion = float((ratio - scale).abs().median() / scale)
    if scale_dispersion > SCALE_DISPERSION_TOL:
        return _fail(
            f"scale dispersion {scale_dispersion:.4f} exceeds "
            f"{SCALE_DISPERSION_TOL} — donor adjustment basis drifts",
            len(overlap),
        )

    # Check 5: post-scale agreement — 95th-pct absolute relative close error.
    agreement_error = float((ratio / scale - 1.0).abs().quantile(0.95))
    if agreement_error > AGREEMENT_TOL:
        return _fail(
            f"post-scale agreement error {agreement_error:.4f} exceeds "
            f"{AGREEMENT_TOL}",
            len(overlap),
        )

    return SeamReport(
        ok=True, scale=scale, offset_hours=offset_hours,
        overlap_bars=int(len(overlap)), scale_dispersion=scale_dispersion,
        agreement_error=agreement_error, reason="",
    )
