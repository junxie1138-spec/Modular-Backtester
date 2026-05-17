"""Build deep hourly OHLCV datasets for the strategy factory.

For each symbol this stitches a Kaggle donor CSV (deep history, manually
placed in data/donor_hourly/) with a yfinance 1h fetch (recent ~730 days)
into one hourly OHLCV CSV under data/raw_hourly/, and emits
data/raw_hourly/_build_report.json as the per-symbol audit trail.

The yfinance fetch is the only network touch; it goes through
_fetch_yfinance_hourly, which calls the _yfinance_download seam unit tests
monkeypatch. Idempotent: re-running refreshes the yfinance tail and
re-validates the seam — that is what makes the Phase B transition just a
matter of scheduling this script.

This is an operator script, NOT a pytest test.

USAGE:
    python -m scripts.build_hourly_dataset                 # the full set
    python -m scripts.build_hourly_dataset --symbols SPY AAPL QQQ
    python -m scripts.build_hourly_dataset --donor-dir data/donor_hourly \\
        --out-dir data/raw_hourly
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.data.hourly_stitch import load_donor, splice, validate_seam
from backtester.data.validators import validate_ohlcv
from backtester.data.yfinance_loader import _normalize_yfinance_frame, _yfinance_download

# The full set of symbols the factory touches (see the design spec's "Symbol
# scope"): SPY (generation) + 14 universe names + 3 promotion tickers + ^VIX.
FACTORY_SYMBOLS: tuple[str, ...] = (
    "SPY", "TSLA", "NVDA", "AMD", "COIN", "GOOGL", "MSTR", "XPEV", "NIO",
    "PLTR", "SMCI", "SHOP", "W", "META", "NFLX", "AAPL", "QQQ", "DIA", "^VIX",
)

# Minimum hourly bar count for `tradable`. ~4 calendar years of regular-session
# hours (252 x 7 x ~4) — below this a WFO batch cannot form enough folds.
# Tunable; documented in the hourly-timeframe design spec.
MIN_HOURLY_BARS = 7000

# yfinance 1h retention is a hard 730-day window — not paginatable.
_YF_HOURLY_PERIOD = "730d"


def _fetch_yfinance_hourly(symbol: str, *, require_volume: bool) -> pd.DataFrame:
    """Fetch regular-session yfinance 1h bars, normalized to the OHLCV contract.

    The result is tz-naive (yfinance returns tz-aware US/Eastern;
    _normalize_yfinance_frame strips the label) with lowercase OHLCV columns.
    The network call is _yfinance_download — unit tests monkeypatch it.
    """
    raw = _yfinance_download(
        symbol, auto_adjust=True, period=_YF_HOURLY_PERIOD, progress=False,
        interval="1h", prepost=False,
    )
    return _normalize_yfinance_frame(raw, require_volume=require_volume)


def build_symbol(symbol: str, *, donor_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Build one symbol's hourly CSV and return its _build_report.json entry.

    Per-symbol failure is never fatal: a yfinance fetch failure yields a
    `failed` entry; a rejected seam falls back to yfinance-only; the caller
    (main) keeps going. Classification (`tradable` / `insufficient_history`)
    is the MIN_HOURLY_BARS policy gate, applied here — the stitcher itself
    makes no tradability claim.
    """
    require_volume = symbol != "^VIX"
    entry: dict[str, Any] = {"symbol": symbol}

    # The recent half: yfinance 1h. Always fetched — it is the live edge.
    try:
        recent = _fetch_yfinance_hourly(symbol, require_volume=require_volume)
    except Exception as exc:  # network / yfinance failure
        entry.update(
            source="failed", validation=f"yfinance fetch failed: {exc}",
            bar_count=0, classification="insufficient_history",
        )
        return entry

    donor_path = donor_dir / f"{symbol}.csv"
    if donor_path.exists():
        try:
            donor = load_donor(donor_path)
            report = validate_seam(donor, recent)
            seam_error = report.reason
        except Exception as exc:  # malformed donor CSV
            donor = None
            report = None
            seam_error = str(exc)
        if report is not None and report.ok:
            aligned = donor.copy()
            if report.offset_hours != 0:
                aligned.index = aligned.index + pd.Timedelta(hours=report.offset_hours)
            seam_ts = recent.index[0]
            output = splice(aligned, recent, seam_ts, report.scale)
            entry.update(
                source="stitched",
                donor_start=str(donor.index.min().date()),
                donor_end=str(donor.index.max().date()),
                seam_date=str(seam_ts.date()),
                scale=round(report.scale, 6),
                seam_offset_hours=report.offset_hours,
                validation="ok",
            )
        else:
            output = recent
            entry.update(
                source="yfinance_only",
                validation=f"seam rejected: {seam_error}",
            )
    else:
        output = recent
        entry.update(source="yfinance_only", validation="no donor CSV")

    # Output integrity gate, then write.
    try:
        validate_ohlcv(output, strict_volume=require_volume)
    except Exception as exc:
        entry.update(
            source="failed", validation=f"output invalid: {exc}",
            bar_count=int(len(output)), classification="insufficient_history",
        )
        return entry

    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / f"{symbol}.csv", index_label="timestamp")

    bar_count = int(len(output))
    entry.update(
        bar_count=bar_count,
        date_start=str(output.index.min().date()),
        date_end=str(output.index.max().date()),
        classification=(
            "tradable" if bar_count >= MIN_HOURLY_BARS else "insufficient_history"
        ),
    )
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        "scripts.build_hourly_dataset",
        description="Build deep hourly OHLCV datasets for the strategy factory.",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=list(FACTORY_SYMBOLS),
        help="symbols to build (default: the full factory set)",
    )
    parser.add_argument(
        "--donor-dir", type=Path, default=Path("data/donor_hourly"),
        help="directory of manually-placed Kaggle donor CSVs",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw_hourly"),
        help="output directory for built hourly CSVs + _build_report.json",
    )
    args = parser.parse_args(argv)

    symbols: dict[str, Any] = {}
    for sym in args.symbols:
        print(f"  building {sym} ...", flush=True)
        entry = build_symbol(sym, donor_dir=args.donor_dir, out_dir=args.out_dir)
        symbols[sym] = entry
        print(
            f"    {entry.get('source')} / {entry.get('classification')} "
            f"({entry.get('bar_count')} bars)",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "min_hourly_bars": MIN_HOURLY_BARS,
        "symbols": symbols,
    }
    (args.out_dir / "_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )

    failed = [s for s, e in symbols.items() if e.get("source") == "failed"]
    insufficient = [
        s for s, e in symbols.items()
        if e.get("classification") == "insufficient_history"
    ]
    print(
        f"\n  build complete: {len(symbols)} symbols, {len(failed)} failed, "
        f"{len(insufficient)} insufficient_history",
        flush=True,
    )
    if symbols.get("SPY", {}).get("classification") != "tradable":
        print(
            "  WARNING: SPY is not tradable — a factory hourly batch will be "
            "refused by preflight.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
