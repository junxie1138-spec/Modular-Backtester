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

| Value | Meaning                                                                 |
|-------|-------------------------------------------------------------------------|
| `1`   | Target long position                                                    |
| `0`   | Target flat                                                             |
| `-1`  | Target short position (requires `execution.allow_short: true` in config)|

- Signals are typically shifted by one bar (`signal.shift(1)`) so the
  engine fills the order on the **next** bar's open, not the current
  bar's close.
- An optional `size` column scales the percent-equity allocation
  (multiplicative with `portfolio.size`). It applies to both long and
  short legs.
- An optional `price_column` (referenced by `SignalFrame.price_column`)
  turns the order into a LIMIT order at that price on the next bar.
  LIMIT is honored only when entering from a flat position (flat → long
  or flat → short). Flips through zero (long → short, short → long) and
  exits to flat are always MARKET.
- Trailing stops are **execution-layer**, not strategy-layer. Configure
  via `execution.trailing_stop_pct` or `execution.trailing_stop_atr_mult`
  in the run YAML. Strategies have no `stop_column` and cannot emit
  per-trade stop levels in v0.3.0. The trailing stop trails the running
  peak (long) or trough (short) since entry and fires as a STOP order on
  the bar after the peak/trough is breached by the configured distance.
  Stop-out exits take precedence over the strategy signal on the same
  bar; the next bar's signal is read normally.
- A strategy that emits only `{0, 1}` continues to work unchanged and
  does not require `allow_short`.
- A strategy that emits `-1` while `execution.allow_short` is `false`
  causes the portfolio simulator to raise `ShortNotAllowedError` at the
  first short signal. Strategy authors should document the requirement
  in their class docstring (see `strategies/rsi_long_short.py`).

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

## v0.4.0 additions (opt-in)

A strategy can opt into the v0.4.0 multi-symbol contract by setting two class
attributes:

```python
class MyStrategy(BaseStrategy[MyParams]):
    uses_multi_symbol = True   # routes through MultiSymbolPortfolioSimulator
    uses_per_bar = True        # signal_for_bar(...) called per (symbol, bar)
```

When `uses_per_bar` is True, the strategy implements:

```python
def signal_for_bar(
    self,
    *,
    symbol: str,
    bar_idx: int,
    data_panel: dict[str, pd.DataFrame],
    indicators_panel: dict[str, pd.DataFrame],
    ctx: StrategyContext,
    params: MyParams,
) -> float:
    """Return target_position in [-1.0, 1.0]. Fractional values are
    interpreted as partial positions (0.5 = half size)."""
```

The strategy may read:
- `ctx.position_phase[symbol]` — a `TSPhase` value (`HARD`, `RUNNER`, or `DISARMED`)
- `ctx.bars_in_phase[symbol]` — bars spent in the current phase
- `ctx.recent_pnl` — rolling portfolio PnL series
- `ctx.regime` — a `RegimeState` with `book_flat`

All four fields are populated by the simulator after the just-processed bar's
state finalizes; strategy decisions for bar `t+1` see state from bar `t`.

Auxiliary OHLCV data (e.g., SPY, ^VIX for regime gates) lives in `data_panel`
under the aux symbol keys declared in `data.aux_symbols`. Aux symbols are not
iterated over for entries — they exist purely as input to indicators and
regime evaluation.

Regime gates (SPY 200-EMA, VIX hysteresis, strategy circuit breaker) live
in the simulator, not the strategy. The strategy reads `ctx.regime.book_flat`
for diagnostics; when True, the simulator forces all positions flat regardless
of the strategy's emitted target.

The v0.3.0 contract (single-symbol, signal ∈ {-1, 0, 1}, no aux_data) is
unchanged. Strategies that do not set `uses_multi_symbol` continue to run
through the original `PortfolioSimulator` path.
