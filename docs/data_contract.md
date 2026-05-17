# Data contract

OHLCV files live in `data/raw/{SYMBOL}.csv` or `data/raw/{SYMBOL}.parquet`.

## Schema

| Column   | Type     | Notes                            |
|----------|----------|----------------------------------|
| date     | date     | Index column; sorted ascending   |
| open     | float    | > 0                              |
| high     | float    | > 0, >= open and >= close        |
| low      | float    | > 0, <= open and <= close        |
| close    | float    | > 0                              |
| volume   | int/float| >= 0                             |

Column names are normalized to lowercase by the loader.

## Invariants

- No NaN values in price or volume columns.
- `DatetimeIndex` is strictly monotonic increasing.
- No duplicate timestamps.

## Loading

```python
from backtester.data.loader import load_symbol
df = load_symbol("SPY", source="csv", root="data/raw", start="2020-01-01", end="2024-01-01")
```

## Validation

```python
from backtester.data.validators import validate_ohlcv
validate_ohlcv(df)  # raises DataError on any violation
```

## Hourly datasets (`data/raw_hourly/`)

`data/raw_hourly/{SYMBOL}.csv` holds the same OHLCV schema as `data/raw/`,
but on an hourly index (tz-naive US/Eastern, regular session, 7 bars/day
open-stamped 09:30..15:30). It is read by the same `CSVDataLoader` — the
loader is unchanged.

These files are *built*, not fetched directly. `scripts/build_hourly_dataset.py`
stitches a deep-history Kaggle donor CSV (`data/donor_hourly/{SYMBOL}.csv`,
manually placed, unverified provenance) with a recent yfinance `1h` fetch:

```bash
python -m scripts.build_hourly_dataset --symbols SPY AAPL QQQ
```

| Path | Role |
|------|------|
| `data/donor_hourly/{SYMBOL}.csv` | Kaggle donor CSVs — deep history, manually placed. |
| `data/raw_hourly/{SYMBOL}.csv` | Built hourly OHLCV output. |
| `data/raw_hourly/_build_report.json` | Per-symbol audit trail: `source` (`stitched` / `yfinance_only` / `failed`), donor date range, seam date, robust scale factor, validation verdict, bar count, date span, and the tradability `classification` (`tradable` / `insufficient_history`). |

A symbol with fewer than `MIN_HOURLY_BARS` (7000) built bars is classified
`insufficient_history` — a thin yfinance-only fallback is not automatically
tradable.
