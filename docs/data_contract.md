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
