from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtester.engine.atr import compute_atr


_SECTOR_MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "sector_map.csv"


def _load_sector_map() -> dict[str, str]:
    if not _SECTOR_MAP_PATH.exists():
        return {}
    with _SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        return {row["symbol"]: row["sector"] for row in csv.DictReader(f)}


def compute_metrics(data: pd.DataFrame) -> dict[str, float]:
    """Compute range/ATR, 200d slope, R^2 for one ticker."""
    close = data["close"]

    def _range(window: np.ndarray) -> float:
        return float(np.percentile(window, 90) - np.percentile(window, 10))

    range_series = close.rolling(63).apply(_range, raw=True)
    range_med = float(range_series.median())

    atr_series = compute_atr(data, 20)
    atr_med = float(atr_series.median())
    range_atr_ratio = range_med / atr_med if atr_med > 0 else 0.0

    log_close = np.log(close)

    def _slope_r2(window: np.ndarray) -> tuple[float, float]:
        x = np.arange(len(window), dtype=float)
        if len(x) < 2:
            return (0.0, 0.0)
        cov = np.cov(x, window, bias=True)[0, 1]
        var_x = float(np.var(x))
        var_y = float(np.var(window))
        if var_x <= 0 or var_y <= 0:
            return (0.0, 0.0)
        slope = cov / var_x
        r = cov / math.sqrt(var_x * var_y)
        return (slope, r * r)

    slopes = []
    r2s = []
    for i in range(199, len(log_close)):
        window = log_close.iloc[i - 199: i + 1].to_numpy()
        s, r2 = _slope_r2(window)
        slopes.append(s)
        r2s.append(r2)
    slope_log_med = float(np.median(slopes)) if slopes else 0.0
    r_squared_med = float(np.median(r2s)) if r2s else 0.0

    return {
        "range_atr_ratio": range_atr_ratio,
        "slope_200d_pct_per_day": float(np.expm1(slope_log_med)),
        "r_squared_200d": r_squared_med,
    }


def passes_filters(
    *,
    range_atr_ratio: float,
    slope_200d_pct_per_day: float,
    r_squared_200d: float,
    min_data_length_ok: bool,
) -> bool:
    if not min_data_length_ok:
        return False
    if abs(slope_200d_pct_per_day) > 0.002 and r_squared_200d > 0.4:
        return False
    if range_atr_ratio < 5.0:
        return False
    return True


def filter_and_rank(
    metrics_by_symbol: dict[str, dict],
    *,
    top: int,
) -> dict[str, dict]:
    kept = {
        sym: m for sym, m in metrics_by_symbol.items()
        if passes_filters(
            range_atr_ratio=m["range_atr_ratio"],
            slope_200d_pct_per_day=m["slope_200d_pct_per_day"],
            r_squared_200d=m["r_squared_200d"],
            min_data_length_ok=True,
        )
    }
    ranked = sorted(kept.items(), key=lambda kv: -kv[1]["range_atr_ratio"])[:top]
    return dict(ranked)


def write_universe_yaml(
    metrics_by_symbol: dict[str, dict],
    *,
    out: Path,
    screening_window: tuple[str, str],
) -> None:
    sector_map = _load_sector_map()
    out_doc = {
        "_meta": {
            "generated_by": "scripts/screen_universe.py",
            "screening_window_start": screening_window[0],
            "screening_window_end": screening_window[1],
            "filters": "|slope| < 0.2%/d AND R^2 < 0.4; range/atr >= 5.0",
        },
        "universe": {},
    }
    for sym, m in metrics_by_symbol.items():
        sector = m.get("sector") or sector_map.get(sym) or "unknown"
        if sector == "unknown":
            print(f"WARNING: unknown sector for {sym}", file=sys.stderr)
        out_doc["universe"][sym] = {
            "sector": sector,
            "range_atr_ratio": round(m["range_atr_ratio"], 2),
            "slope_200d": round(m["slope_200d_pct_per_day"], 4),
            "r_squared_200d": round(m["r_squared_200d"], 3),
        }
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_doc, f, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("screen_universe")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--data-root", default="data/raw")
    args = parser.parse_args(argv)

    from backtester.data.loader import load_symbol

    candidates = [
        s.strip() for s in args.candidates.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]

    sector_map = _load_sector_map()
    metrics_by_symbol: dict[str, dict] = {}
    for sym in candidates:
        try:
            data = load_symbol(
                symbol=sym, source="yfinance", root=args.data_root,
                start=args.start, end=args.end,
                require_volume=False,
            )
        except Exception as exc:
            print(f"WARNING: skipping {sym}: {exc}", file=sys.stderr)
            continue
        if len(data) < 504:
            print(f"WARNING: {sym} has only {len(data)} bars; skipping", file=sys.stderr)
            continue
        m = compute_metrics(data)
        m["sector"] = sector_map.get(sym, "unknown")
        metrics_by_symbol[sym] = m

    ranked = filter_and_rank(metrics_by_symbol, top=args.top)
    write_universe_yaml(ranked, out=args.out, screening_window=(args.start, args.end))
    print(f"wrote {len(ranked)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
