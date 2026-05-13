# Strategy contract

A strategy is exactly one Python file under `strategies/` that defines:

1. One `@dataclass(slots=True)` for its parameters.
2. One class that inherits from `BaseStrategy[ParamsType]`.

Both must live in the same module. The strategy must be registered in
`backtester/strategies/registry.py` exactly once.

## Required imports

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy
```

## Required class attributes

| Attribute     | Type  | Notes                                          |
|---------------|-------|------------------------------------------------|
| `strategy_id` | str   | Unique snake_case identifier                   |
| `version`     | str   | Semver-like string                             |
| `asset_type`  | str   | `"stock"` for MVP                              |
| `timeframe`   | str   | `"1d"` for daily                               |

## Required methods

```python
@classmethod
def params_type(cls) -> type: ...

def warmup_bars(self, params) -> int: ...           # optional, default 0

def indicators(self, data, params) -> pd.DataFrame: ...

def generate_signals(self, data, indicators, ctx, params) -> SignalFrame: ...
```

## Data assumptions

- Index is a `DatetimeIndex`, sorted ascending, no duplicates.
- Columns include lowercase `open`, `high`, `low`, `close`, `volume`.
- Prices are positive, no NaNs in price columns.
- Only past and current bars may be used. **Never read future rows.**

## Signal semantics

- `1` = target long position.
- `0` = target flat.
- Signals are typically shifted by one bar (`signal.shift(1)`) so the
  engine fills the order on the **next** bar's open, not the current
  bar's close.
- An optional `size` column scales the percent-equity allocation
  (multiplicative with `portfolio.size`).
- An optional `price_column` (referenced by `SignalFrame.price_column`)
  turns the order into a LIMIT order at that price on the next bar.

## Rules for AI-generated strategies

1. Exactly one params dataclass.
2. Exactly one public strategy class.
3. `strategy_id` in snake_case, globally unique.
4. No file, network, or env access.
5. Import only `BaseStrategy`, `SignalFrame`, `StrategyContext` from the framework.
6. Use only past and current bars.
7. Shift tradable signals by one bar unless explicitly using same-bar execution.
8. Keep helper logic in the same file unless reused across 3+ strategies.

## Minimal example

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FlatParams:
    size: float = 1.0


class FlatStrategy(BaseStrategy[FlatParams]):
    strategy_id = "flat"

    @classmethod
    def params_type(cls):
        return FlatParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params: FlatParams):
        df = pd.DataFrame({"signal": 0, "size": params.size}, index=data.index)
        return SignalFrame(data=df)
```
