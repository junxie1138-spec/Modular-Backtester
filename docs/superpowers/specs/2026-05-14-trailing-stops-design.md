# Trailing-stop support — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Add an execution-layer trailing stop to the Modular Stock Backtester. Distance is configurable as either a percentage of the running peak (long) / trough (short) OR a multiple of recent ATR. Stops trail every bar from entry onward and fire as exit fills, taking priority over the strategy's signal. Default is disabled — every existing config and strategy produces byte-identical results to v0.2.0.

**Version target:** New feature for the v0.3.x line. Bump `pyproject.toml` from `0.2.0` to `0.3.0` after acceptance.

---

## 1. Behavior

### 1.1 Terminology

- **Trailing-stop state** = `(peak_high, trough_low, armed)` plus the active distance spec.
- **Armed** = the position is non-flat AND the trailing-stop feature is enabled. Disarmed when the position goes flat (regardless of cause) and re-armed on a fresh entry/flip.
- **Stop level** at end of bar `i`:
  - Long (sign +1): `max(peak_high since entry) - distance(i)`. For percent mode: `peak_high * (1 - trailing_stop_pct)`. For ATR mode: `peak_high - trailing_stop_atr_mult * ATR[i]`.
  - Short (sign -1): mirror image — `trough_low + distance(i)`.
- **Fire** = the next bar's OHLC triggers the STOP order at the FillEngine (long: `low <= stop_price`; short: `high >= stop_price`).

### 1.2 Per-bar state machine

For each bar `i` (0-indexed over `data.index`), the simulator does the following in order. `prev_sign` is `_sign(pos.qty)` at the start of bar `i`, before any orders fill. `ts` is the singleton `TrailingStopState` instance.

1. **Execute pending stop order.** If `pending_stop` is non-None, submit it to the broker. If it fills:
   - Tag the resulting Fill with `reason="trailing_stop"`.
   - Apply the fill to the position. `pos.qty` becomes 0 (the stop always closes the full position).
   - Set `stop_filled := True` and **clear `pending_stop`**.
   - Mark `ts.armed = False`.
2. **Execute pending signal order.** If `pending_signal` is non-None:
   - If `stop_filled` is True → discard `pending_signal` (cancelled by the same-bar stop hit).
   - Otherwise → submit, tag the Fill with `reason="signal"`, apply to position.
   - Clear `pending_signal`.
3. **Compute `new_sign`** = `_sign(pos.qty)` after the fills.
4. **Reset trailing state on sign change.** If `new_sign != prev_sign`:
   - If `new_sign != 0` → entry from flat OR flip through zero. Call `ts.reset(entry_price=last_fill.price)`. This sets `peak_high = trough_low = entry_price` and `armed = True`.
   - If `new_sign == 0` and `not stop_filled` → signal-driven exit. Call `ts.disarm()`.
   - If `new_sign == 0` and `stop_filled` → already disarmed in step 1.
5. **Update peak/trough with this bar's OHLC.** If `ts.armed`: `ts.peak_high = max(ts.peak_high, bar.high)` and `ts.trough_low = min(ts.trough_low, bar.low)`.
6. **Read this bar's signal** `sig = signals[sig_col].iloc[i]`. If `sig == -1` and not `broker.allow_short`, raise `ShortNotAllowedError` (unchanged from v0.2.0).
7. **Schedule orders for bar i+1** (only if `i + 1 < len(index)`):
   - **Stop order**: If `ts.enabled` and `new_sign != 0` and `ts.armed`, compute `stop_px = ts.stop_price(new_sign, bar_idx=i)`. If `stop_px is not None` (ATR mode may return None during warmup), build a `STOP` order for the full position quantity (`abs(pos.qty)`) on the opposing side, with `stop_price=stop_px`. Store in `pending_stop`.
   - **Signal order**: Same logic as the v0.2.0 simulator, but using `new_sign` (the post-fill position sign) instead of pre-fill. Store in `pending_signal`. LIMIT semantics from v0.2.0 are unchanged.
8. **Mark to market** at `bar.close` (unchanged).

### 1.3 Precedence summary

- **Stop wins over signal, same bar.** If both pending orders are present and the stop fills, the signal order is dropped. The trade row carries `reason="trailing_stop"`.
- **Stop fills cancel only that bar's signal order.** The next bar reads its own signal normally; if the signal still wants an entry, the simulator schedules a fresh entry order from flat.
- **Stop never overrides flat-to-non-flat transitions.** It can only fire while the position is non-flat. The entry itself is always signal-driven.

### 1.4 Reset semantics on every position change

- **flat → long**: `ts.reset(fill.price)`. `peak_high = trough_low = entry_price`.
- **flat → short**: `ts.reset(fill.price)`. Same call; the `new_sign` argument used by `stop_price()` determines whether peak or trough drives the level.
- **long → flat (signal)**: `ts.disarm()`.
- **short → flat (signal)**: `ts.disarm()`.
- **long → flat (stop)**: `ts.disarm()` is called inside step 1.
- **short → flat (stop)**: same.
- **long → short (combined flip)**: at end-of-bar (step 4), `new_sign = -1 != prev_sign = +1`. `ts.reset(fill.price)` is called with the flip-fill price. The new short leg trails from the flip price, not from any prior peak.
- **short → long (combined flip)**: symmetric.

### 1.5 Backwards compatibility

When BOTH `trailing_stop_pct is None` AND `trailing_stop_atr_mult is None`:
- `TrailingStopState.enabled` returns False.
- `pending_stop` is always `None`.
- The simulator's step 1 (execute pending stop) is a no-op.
- Step 7's stop-order branch is skipped.
- Behavior is byte-identical to v0.2.0 — verified by re-running `configs/backtests/sma_cross_spy.yaml` and comparing `summary.json` numerics.

### 1.6 ATR computation

`ATR[i] = SMA(TR, period)[i]`, where `TR[i] = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)`. Uses standard pandas `rolling(period).mean()` semantics — first defined value is at bar `period`. Bars where `ATR[i]` is NaN return `stop_price = None` from `TrailingStopState.stop_price`, and the simulator skips scheduling a STOP order for that bar (the position remains held, no stop guard active until ATR is valid).

`TR[0]` uses `close[-1]` which is undefined; convention: `TR[0] = high[0] - low[0]`. (Standard pandas idiom: `.fillna({...})` after the diff is taken.)

ATR is pre-computed once at the start of `PortfolioSimulator.simulate` over the full `data` frame and stored on the `TrailingStopState` instance as a `pd.Series` aligned to `data.index`.

---

## 2. Configuration

### 2.1 New `ExecutionConfig` fields

```python
@dataclass(slots=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_fractional: bool = False
    allow_short: bool = False
    # New in v0.3.0:
    trailing_stop_pct: Optional[float] = None
    trailing_stop_atr_mult: Optional[float] = None
    trailing_stop_atr_period: int = 14
```

### 2.2 Field semantics

| Field | Default | Constraint |
|---|---|---|
| `trailing_stop_pct` | `None` | If set, must be `> 0` and `< 1`. Stop distance is `trailing_stop_pct * peak_or_trough`. |
| `trailing_stop_atr_mult` | `None` | If set, must be `> 0`. Stop distance is `trailing_stop_atr_mult * ATR`. |
| `trailing_stop_atr_period` | `14` | Must be `>= 2`. Ignored when `trailing_stop_atr_mult is None`. |

### 2.3 Validation (in `backtester/config/validation.py`)

All validation rules raise `backtester.core.exceptions.ConfigError` (consistent with the file's existing style).

- `trailing_stop_pct` and `trailing_stop_atr_mult` are mutually exclusive — at most one may be non-None. Both set raises `ConfigError("execution.trailing_stop_pct and trailing_stop_atr_mult are mutually exclusive")`.
- If `trailing_stop_pct is not None`: must be `0 < pct < 1`, else `ConfigError("execution.trailing_stop_pct must be in (0, 1)")`.
- If `trailing_stop_atr_mult is not None`: must be `> 0`, else `ConfigError("execution.trailing_stop_atr_mult must be > 0")`.
- `trailing_stop_atr_period` must be `>= 2`, else `ConfigError("execution.trailing_stop_atr_period must be >= 2")`.

### 2.4 YAML sample

```yaml
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
  allow_short: false
  trailing_stop_pct: 0.05    # 5% trailing stop
```

Or with ATR:

```yaml
execution:
  trailing_stop_atr_mult: 3.0
  trailing_stop_atr_period: 14
```

---

## 3. Reporting

### 3.1 `Fill` dataclass

Add an optional `reason: str = "signal"` field at the end of the dataclass:

```python
@dataclass(slots=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float
    reason: str = "signal"  # NEW
```

Default `"signal"` keeps existing Fill constructors (incl. test helpers) working unchanged. Only the simulator's stop-execution branch sets `reason="trailing_stop"`.

### 3.2 `trades.csv` schema

A new last column `reason` is appended:

```
timestamp,side,qty,price,commission,notional,reason
2024-01-03,buy,95,420.5,3.99,39947.5,signal
2024-01-15,sell,95,438.2,4.16,41629.0,trailing_stop
```

Downstream readers that select by column name are unaffected. The CSV-as-positional reader test (if any) is updated; review confirms there is none.

---

## 4. File layout

| File | Action |
|---|---|
| `backtester/engine/atr.py` | create |
| `backtester/engine/trailing_stop.py` | create |
| `backtester/engine/fills.py` | modify — add `reason: str = "signal"` field |
| `backtester/engine/portfolio.py` | modify — dual pending slots, TS integration, reason tagging on trades_df |
| `backtester/config/models.py` | modify — 3 new ExecutionConfig fields |
| `backtester/config/validation.py` | modify — 4 new validation rules |
| `configs/backtests/sma_cross_spy_trailing.yaml` | create |
| `tests/unit/test_atr.py` | create |
| `tests/unit/test_trailing_stop.py` | create |
| `tests/unit/test_orders_fills.py` | append — new `reason` field test |
| `tests/unit/test_portfolio.py` | append — trailing-stop simulator tests |
| `tests/unit/test_config_models.py` | append — new field defaults |
| `tests/unit/test_config_validation.py` | append — mutual-exclusion + bounds |
| `tests/integration/test_run_backtest_cli.py` | append — trailing-stop CLI smoke + drawdown sanity check |
| `tests/integration/test_backwards_compat.py` | create — sma_cross_spy numerics unchanged |
| `docs/strategy_contract.md` | modify — clarify stops are execution-layer |
| `docs/runbook.md` | modify — Limitations section |
| `README.md` | modify — Execution model section |
| `pyproject.toml` | bump `0.2.0` → `0.3.0` |

No changes to: strategies, `backtester/engine/broker.py` (Broker just submits orders — no policy change), `backtester/engine/orders.py` (Order already supports `stop_price`), `backtester/engine/position.py` (Position is pure fill algebra — trailing-state lives at the simulator layer), runners, analytics, optimizer, splitter, stitcher.

---

## 5. Implementation notes

### 5.1 `TrailingStopState` (in `backtester/engine/trailing_stop.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class TrailingStopState:
    pct: Optional[float] = None
    atr_mult: Optional[float] = None
    atr_series: Optional[pd.Series] = None  # pre-computed; aligned to data.index
    peak_high: float = 0.0
    trough_low: float = float("inf")
    armed: bool = False

    @property
    def enabled(self) -> bool:
        return self.pct is not None or self.atr_mult is not None

    def reset(self, entry_price: float) -> None:
        self.peak_high = entry_price
        self.trough_low = entry_price
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.peak_high = 0.0
        self.trough_low = float("inf")

    def update(self, high: float, low: float) -> None:
        if not self.armed:
            return
        if high > self.peak_high:
            self.peak_high = high
        if low < self.trough_low:
            self.trough_low = low

    def stop_price(self, sign: int, bar_idx: int) -> Optional[float]:
        if not self.armed or sign == 0:
            return None
        if self.pct is not None:
            return self.peak_high * (1.0 - self.pct) if sign > 0 else self.trough_low * (1.0 + self.pct)
        # ATR mode
        atr_val = float(self.atr_series.iloc[bar_idx])  # type: ignore[union-attr]
        if pd.isna(atr_val):
            return None
        return (self.peak_high - self.atr_mult * atr_val) if sign > 0 else (self.trough_low + self.atr_mult * atr_val)
```

### 5.2 `compute_atr` (in `backtester/engine/atr.py`)

```python
from __future__ import annotations
import pandas as pd


def compute_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using SMA smoothing.

    TR[i] = max(high - low, |high - prev_close|, |low - prev_close|)
    TR[0] = high[0] - low[0] (prev_close undefined).
    ATR[i] = rolling SMA of TR over `period` bars.
    """
    if period < 2:
        raise ValueError("ATR period must be >= 2")
    high = data["high"]
    low = data["low"]
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr.rolling(period).mean()
```

### 5.3 Simulator changes (in `backtester/engine/portfolio.py`)

Two pending slots:

```python
pending_signal: Optional[Order] = None
pending_stop: Optional[Order] = None
```

Single `TrailingStopState` instance constructed once:

```python
ts = TrailingStopState(
    pct=broker.config.trailing_stop_pct,
    atr_mult=broker.config.trailing_stop_atr_mult,
    atr_series=(
        compute_atr(data, broker.config.trailing_stop_atr_period)
        if broker.config.trailing_stop_atr_mult is not None else None
    ),
)
```

Per-bar loop follows the eight-step state machine in §1.2 exactly. The signal-order construction branch is unchanged from v0.2.0 except that `prev_sign` is now `new_sign` (after fills).

Trade-row construction picks up the new `reason` field:

```python
trades_df = pd.DataFrame([
    {
        "timestamp": f.timestamp, "side": f.side.value, "qty": f.qty,
        "price": f.price, "commission": f.commission, "notional": f.notional,
        "reason": f.reason,
    } for f in fills
])
```

When `fills` is empty, the empty DataFrame still has zero rows; downstream code that accesses by column name (`trades["side"]`) handles both shapes already — verified by review of `analytics/metrics.py`.

---

## 6. Tests

### 6.1 Unit — `tests/unit/test_atr.py` (4 tests)

1. **`test_atr_first_value_is_high_minus_low`** — verifies the boundary convention.
2. **`test_atr_period_2_matches_hand_calc`** — synthetic 5-bar series, hand-computed expected values.
3. **`test_atr_nan_before_period`** — first `period-1` outputs are NaN.
4. **`test_atr_invalid_period_raises`** — `period=1` raises `ValueError`.

### 6.2 Unit — `tests/unit/test_trailing_stop.py` (10 tests)

1. **`test_disabled_by_default`** — `TrailingStopState().enabled is False`.
2. **`test_enabled_when_pct_set`** — `TrailingStopState(pct=0.05).enabled is True`.
3. **`test_enabled_when_atr_set`** — `TrailingStopState(atr_mult=2.0, atr_series=pd.Series([...])).enabled is True`.
4. **`test_reset_arms_and_sets_peak_trough`** — after `reset(100.0)`, both `peak_high` and `trough_low` equal 100.0, `armed is True`.
5. **`test_update_long_peak_ratchets_up_only`** — sequence of `(high, low) = (101, 95), (98, 92), (105, 100)` → `peak_high` ends at 105, only moves up.
6. **`test_update_short_trough_ratchets_down_only`** — symmetric.
7. **`test_pct_stop_price_long`** — `peak_high=100`, `pct=0.05` → `stop_price(+1, 0) == 95.0`.
8. **`test_pct_stop_price_short`** — `trough_low=100`, `pct=0.05` → `stop_price(-1, 0) == 105.0`.
9. **`test_atr_stop_price_long_uses_indexed_value`** — `atr_series=[NaN, 2.0, 3.0]`, `peak_high=100`, `atr_mult=2.0` → `stop_price(+1, 1) == 96.0`; `stop_price(+1, 0) is None` (NaN ATR).
10. **`test_disarm_clears_state`** — after `disarm()`, `armed is False`, `stop_price` returns None.

### 6.3 Unit — `tests/unit/test_orders_fills.py` (1 new test, append)

1. **`test_fill_reason_defaults_to_signal`** — `Fill(...).reason == "signal"`; override works.

### 6.4 Unit — `tests/unit/test_config_models.py` (3 new tests, append)

1. **`test_execution_config_trailing_stop_pct_defaults_none`**.
2. **`test_execution_config_trailing_stop_atr_mult_defaults_none`**.
3. **`test_execution_config_trailing_stop_atr_period_defaults_14`**.

### 6.5 Unit — `tests/unit/test_config_validation.py` (4 new tests, append; create file if missing)

1. **`test_trailing_stop_pct_and_atr_mutually_exclusive`** — both set raises.
2. **`test_trailing_stop_pct_out_of_range`** — `pct=0`, `pct=1.0`, `pct=-0.1` all raise.
3. **`test_trailing_stop_atr_mult_must_be_positive`** — `atr_mult=0`, `atr_mult=-1` raise.
4. **`test_trailing_stop_atr_period_too_small`** — `atr_period=1` raises.

### 6.6 Unit — `tests/unit/test_portfolio.py` (6 new tests, append)

1. **`test_no_trailing_stop_is_byte_identical_to_v0_2_0`** — run sma_cross-style signals on synthetic data with `trailing_stop_pct=None`; compare equity curve to a baseline run without the new fields.
2. **`test_long_trailing_stop_fires_on_drawdown`** — synthetic uptrend then sharp drop: position long, drop > pct%, expect a stop-out fill with `reason="trailing_stop"`.
3. **`test_short_trailing_stop_fires_on_rally`** — symmetric for short.
4. **`test_gap_through_stop_fills_at_open`** — synthetic data where bar N+1's open is far below the long stop level. Expected fill price ≈ open (the FillEngine's existing STOP semantic). Assert `fill_price < stop_level` and `reason == "trailing_stop"`.
5. **`test_stop_wins_over_signal_flip_same_bar`** — long open, signal flips to -1 the same bar the stop would fire. Expected: one stop-out fill (`reason="trailing_stop"`, side="sell") then a fresh short entry on the FOLLOWING bar (`reason="signal"`, side="sell"). Position goes long → flat → short. Position is NEVER held simultaneously across signs.
6. **`test_stop_resets_on_flip_through_zero`** — long open, signal flips long → short (no stop fires). After the combined-flip fill, the trailing state is re-armed at the flip-fill price for the new short leg. Verified by triggering the short stop and asserting `trough_low` was tracked only from the flip price onward.

### 6.7 Integration — `tests/integration/test_backwards_compat.py` (new file, 1 test)

1. **`test_sma_cross_spy_unchanged_with_trailing_disabled`** — run the bundled `configs/backtests/sma_cross_spy.yaml` config end-to-end. Read the `summary.json`. Compare numeric fields against a hard-coded golden dict captured during plan execution (Task 20). Allowed tolerance: `n_trades` exact integer; floats to within `1e-9` absolute, `1e-12` relative.

### 6.8 Integration — `tests/integration/test_run_backtest_cli.py` (append, 1 test)

1. **`test_run_backtest_cli_trailing_stop_smoke`** — run `configs/backtests/sma_cross_spy_trailing.yaml` (`trailing_stop_pct: 0.05`). Assert exit 0, summary exists, trades.csv exists with a `reason` column, at least one row has `reason="trailing_stop"`, and the run's `max_drawdown` is strictly less negative than the no-stop SMA-Cross baseline run from the same test session. (The drawdown comparison is a soft sanity check.)

### 6.9 Test count

Baseline: 172. New: 4 + 10 + 1 + 3 + 4 + 6 + 1 + 1 = **30** new tests. Target: **202** tests passing.

---

## 7. Config samples

### 7.1 `configs/backtests/sma_cross_spy_trailing.yaml`

```yaml
run_name: sma_cross_spy_trailing
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
  size: 1.0
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: "csv"
  root: "data/raw"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
  allow_short: false
  trailing_stop_pct: 0.05
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
output_root: "output/runs"
```

---

## 8. Out of scope

Deliberately deferred to a future v0.3.x phase or v0.4.0:

- **Fixed-dollar stops.** Same code path as percentage; would add a third mutually-exclusive `trailing_stop_abs` field. Trivial follow-up.
- **Asymmetric long/short distances.** E.g., wider stop on the short side. Requires duplicating config fields.
- **Strategy-emitted stops.** Strategies could in principle emit a per-trade stop in their `SignalFrame`. Out of scope — the brief explicitly places stops at the execution layer.
- **Ratcheting variants.** "Move stop to breakeven once unrealized PnL reaches X%" and similar conditional behaviors.
- **Time-based stops.** "Exit after N bars without progress."
- **Stops on grid / WFO optimization.** Trailing stop parameters are not yet first-class search dimensions in `OptimizationConfig.param_space`. Adding them is a separate, contained feature.
- **Borrow-cost interaction.** Out of scope for the same reason it was in v0.2.0 — handled by a future short-accounting feature.

---

## 9. Acceptance criteria

1. **`python -m pytest -q`** passes with at least **202 tests** (baseline 172 + 30 new), zero regressions in the 172 baseline.
2. **`python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml`** exits 0 and produces a `summary.json` whose `total_return`, `sharpe`, `max_drawdown`, `n_trades`, `final_equity` are byte-identical to the v0.2.0 baseline (captured as a golden in `tests/integration/test_backwards_compat.py`).
3. **`python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy_trailing.yaml`** exits 0, writes a `trades.csv` containing at least one row with `reason="trailing_stop"`, and produces a smaller `max_drawdown` magnitude than the no-stop SMA-Cross run on the same data.
4. **Unit-test surface** covers: long stop ratchets up only, short stop ratchets down only, gap-through-stop exits at open, stop wins over signal same-bar, stop state resets on flip, ATR-mode stop level, ATR warmup returns None, mutually-exclusive config validation.
5. **`pyproject.toml`** version is `0.3.0` and `git tag v0.3.0` is created (push only after user confirmation).
6. **README's "Execution model" section** mentions trailing stops (pct + ATR), and `docs/runbook.md` lists trailing-stop limitations (mutually exclusive with signal-driven exits on same bar; no fixed-dollar; etc.).
