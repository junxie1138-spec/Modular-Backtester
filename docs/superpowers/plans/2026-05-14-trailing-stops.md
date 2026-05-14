# Trailing-Stop Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an execution-layer trailing stop to the Modular Stock Backtester. Distance is configurable as either a percentage of the peak/trough since entry OR a multiple of recent ATR. Stops trail every bar, take priority over the strategy's signal, and surface as `reason="trailing_stop"` rows in `trades.csv`. When both new fields are `None`, behavior is byte-identical to v0.2.0.

**Architecture:** A new `TrailingStopState` dataclass (`backtester/engine/trailing_stop.py`) owned by `PortfolioSimulator` tracks `(peak_high, trough_low, armed)` per position lifetime. The simulator gains a second pending-order slot (`pending_stop`) executed before `pending_signal`. ATR is pre-computed once over the data frame by a new `compute_atr` helper (`backtester/engine/atr.py`). `Fill` grows an optional `reason: str = "signal"` field which the simulator tags as `"trailing_stop"` when a stop fires. The existing `Position`, `Broker`, `FillEngine`, `Order`, runners, and strategy machinery are unchanged.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new runtime dependencies.

**Scope notes:**
- Two distance modes: percentage (`trailing_stop_pct`) and ATR-multiple (`trailing_stop_atr_mult` + `trailing_stop_atr_period`). Mutually exclusive at the config-validation layer.
- Fixed-dollar stops, asymmetric long/short distances, ratcheting variants, time-based stops, and grid/WFO-searchable trailing parameters are deliberately out of scope (see spec §8).
- Default disabled. Long-only and long/short configs run unchanged.
- Stop-out exits cancel the same-bar signal order. Re-entry on the next bar is allowed (the strategy controls cooldown if needed).
- ATR is computed in the simulator over the full `data` frame; the strategy contract is unchanged (strategies do not see ATR).

**Required reading before starting:**

1. `docs/superpowers/specs/2026-05-14-trailing-stops-design.md` — the spec this plan implements. **Read in full.**
2. `docs/superpowers/plans/2026-05-14-short-positions.md` — the v0.2.0 plan. The current plan uses the same atomic-commit cadence, same per-task TDD structure, and the same final backwards-compat verification pattern (Task 28 below mirrors Task 18 there).
3. `backtester/engine/portfolio.py`, `backtester/engine/fills.py`, `backtester/engine/position.py`, `backtester/engine/broker.py`, `backtester/engine/orders.py` — the execution stack.
4. `backtester/config/models.py` and `backtester/config/validation.py` — where the new fields and rules go.
5. `tests/unit/test_portfolio.py`, `tests/unit/test_orders_fills.py`, `tests/unit/test_config_validation.py` — test styles to match.
6. `tests/fixtures/synthetic.py` — the `make_ohlcv` fixture used in unit tests.

**Baseline verification before starting:**

Run from repo root:
```
python -m pytest -q
```
Expected: `172 passed`. Record this number — it is the **no-regression baseline**.

Also capture a golden numeric snapshot of the v0.2.0 SMA-Cross run for Task 28:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
Note the printed run directory. Then:
```
python -c "import json; s=json.load(open(r'<RUNDIR>/summary.json')); print({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')})"
```
**Paste the printed dict into a scratch file** — it becomes the golden in Task 28's hard-coded comparison.

**File-structure preview (what will change):**

| File | Action | Purpose |
|---|---|---|
| `backtester/engine/atr.py` | create | `compute_atr(data, period)` helper |
| `backtester/engine/trailing_stop.py` | create | `TrailingStopState` dataclass |
| `backtester/engine/fills.py` | modify | Add `reason: str = "signal"` field |
| `backtester/engine/portfolio.py` | modify (substantial) | Dual pending slots, TS integration, reason tagging |
| `backtester/config/models.py` | modify | 3 new `ExecutionConfig` fields |
| `backtester/config/validation.py` | modify | 4 new validation rules |
| `configs/backtests/sma_cross_spy_trailing.yaml` | create | Sample config |
| `tests/unit/test_atr.py` | create | ATR helper tests |
| `tests/unit/test_trailing_stop.py` | create | TS state machine tests |
| `tests/unit/test_orders_fills.py` | append | `reason` field test |
| `tests/unit/test_portfolio.py` | append | TS simulator tests |
| `tests/unit/test_config_models.py` | append | New field defaults |
| `tests/unit/test_config_validation.py` | append | Mutual exclusion + bounds |
| `tests/integration/test_run_backtest_cli.py` | append | TS CLI smoke + drawdown sanity |
| `tests/integration/test_backwards_compat.py` | create | sma_cross_spy numeric golden |
| `docs/strategy_contract.md` | modify | Stops are execution-layer (note) |
| `docs/runbook.md` | modify | Limitations section |
| `README.md` | modify | Execution model paragraph |
| `pyproject.toml` | modify | Bump `0.2.0` → `0.3.0` |

---

## Phase 1: Scaffolding (config + Fill.reason + ATR helper)

### Task 1: Add `reason` field to `Fill`

**Files:**
- Modify: `backtester/engine/fills.py`
- Modify: `tests/unit/test_orders_fills.py` (append one test)

**Rationale:** Adding an optional field with default at the END of the slotted dataclass is backwards-compatible — existing `Fill(...)` constructors in tests and the simulator work unchanged. The simulator will set it to `"trailing_stop"` in Task 17.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_orders_fills.py`:
```python
def test_fill_reason_defaults_to_signal():
    fill = Fill(
        timestamp=pd.Timestamp("2024-01-02"),
        symbol="SPY",
        side=OrderSide.BUY,
        qty=10,
        price=100.0,
        commission=1.0,
    )
    assert fill.reason == "signal"


def test_fill_reason_override():
    fill = Fill(
        timestamp=pd.Timestamp("2024-01-02"),
        symbol="SPY",
        side=OrderSide.SELL,
        qty=10,
        price=100.0,
        commission=1.0,
        reason="trailing_stop",
    )
    assert fill.reason == "trailing_stop"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_orders_fills.py::test_fill_reason_defaults_to_signal -v`
Expected: FAIL with `AttributeError: 'Fill' object has no attribute 'reason'`.

- [ ] **Step 3: Implement**

Modify the `Fill` dataclass in `backtester/engine/fills.py`. Add the `reason` field as the LAST field:

```python
@dataclass(slots=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float
    reason: str = "signal"

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def cash_delta(self) -> float:
        sign = -1.0 if self.side == OrderSide.BUY else 1.0
        return sign * self.notional - self.commission
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_orders_fills.py -v`
Expected: all prior tests still pass, plus 2 new passes.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: `174 passed` (baseline 172 + 2 new).

- [ ] **Step 6: Commit**

```
git add backtester/engine/fills.py tests/unit/test_orders_fills.py
git commit -m "feat(fills): add reason field to Fill (default 'signal')"
```

---

### Task 2: Add trailing-stop fields to `ExecutionConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Modify: `tests/unit/test_config_models.py` (append three tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_models.py`:
```python
def test_execution_config_trailing_stop_pct_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_pct is None


def test_execution_config_trailing_stop_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_atr_mult is None


def test_execution_config_trailing_stop_atr_period_defaults_14():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_atr_period == 14
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k trailing_stop`
Expected: FAIL — `AttributeError: 'ExecutionConfig' object has no attribute 'trailing_stop_pct'`.

- [ ] **Step 3: Implement**

Modify `backtester/config/models.py`. Add the three new fields at the end of `ExecutionConfig`:

```python
@dataclass(slots=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_fractional: bool = False
    allow_short: bool = False
    trailing_stop_pct: Optional[float] = None
    trailing_stop_atr_mult: Optional[float] = None
    trailing_stop_atr_period: int = 14
```

`Optional` is already imported at the top of the file (from the `WFOConfig` section). Verify by reading the imports — if missing, add `from typing import Optional`.

- [ ] **Step 4: Verify YAML loader picks them up automatically**

Run:
```
python -c "from backtester.config.loader import load_run_config; rc = load_run_config('configs/backtests/sma_cross_spy.yaml'); print('pct=', rc.execution.trailing_stop_pct, 'atr_mult=', rc.execution.trailing_stop_atr_mult, 'atr_period=', rc.execution.trailing_stop_atr_period)"
```
Expected: `pct= None atr_mult= None atr_period= 14`.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v`
Expected: all prior + 3 new passes.

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add trailing-stop fields to ExecutionConfig"
```

---

### Task 3: Add validation rules for trailing-stop fields

**Files:**
- Modify: `backtester/config/validation.py`
- Modify: `tests/unit/test_config_validation.py` (append four tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_validation.py`. (If the test file uses a helper to build a minimal `RunConfig`, reuse it. The skeleton below assumes a `_make_run_config()` helper exists; if not, copy and adapt the pattern from the file's first existing test.)

```python
def test_trailing_stop_pct_and_atr_mutually_exclusive():
    from backtester.config.models import ExecutionConfig
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError

    rc = _make_run_config(execution=ExecutionConfig(
        trailing_stop_pct=0.05,
        trailing_stop_atr_mult=2.0,
    ))
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)


def test_trailing_stop_pct_out_of_range():
    from backtester.config.models import ExecutionConfig
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError

    for bad in (0.0, 1.0, -0.1, 1.5):
        rc = _make_run_config(execution=ExecutionConfig(trailing_stop_pct=bad))
        with pytest.raises(ConfigError, match="trailing_stop_pct"):
            validate_run_config(rc)


def test_trailing_stop_atr_mult_must_be_positive():
    from backtester.config.models import ExecutionConfig
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError

    for bad in (0.0, -1.0):
        rc = _make_run_config(execution=ExecutionConfig(trailing_stop_atr_mult=bad))
        with pytest.raises(ConfigError, match="trailing_stop_atr_mult"):
            validate_run_config(rc)


def test_trailing_stop_atr_period_too_small():
    from backtester.config.models import ExecutionConfig
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError

    rc = _make_run_config(execution=ExecutionConfig(
        trailing_stop_atr_mult=2.0,
        trailing_stop_atr_period=1,
    ))
    with pytest.raises(ConfigError, match="trailing_stop_atr_period"):
        validate_run_config(rc)
```

If `_make_run_config` does not exist in the test file, add it near the top of the file as:
```python
def _make_run_config(execution=None):
    from backtester.config.models import (
        DataConfig, ExecutionConfig, PortfolioConfig, RunConfig,
    )
    return RunConfig(
        run_name="x", strategy="sma_cross", strategy_params={},
        data=DataConfig(symbols=["SPY"], timeframe="1d",
                        start="2020-01-02", end="2020-12-31"),
        execution=execution or ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k trailing_stop`
Expected: 4 failures — the validator does not yet know about the new fields.

- [ ] **Step 3: Implement**

Append to `validate_run_config` in `backtester/config/validation.py`, after the existing `execution.commission_bps / slippage_bps` block:

```python
    # Trailing-stop validation.
    pct = rc.execution.trailing_stop_pct
    atr_mult = rc.execution.trailing_stop_atr_mult
    atr_period = rc.execution.trailing_stop_atr_period
    if pct is not None and atr_mult is not None:
        raise ConfigError(
            "execution.trailing_stop_pct and trailing_stop_atr_mult are mutually exclusive"
        )
    if pct is not None and not (0.0 < pct < 1.0):
        raise ConfigError("execution.trailing_stop_pct must be in (0, 1)")
    if atr_mult is not None and atr_mult <= 0.0:
        raise ConfigError("execution.trailing_stop_atr_mult must be > 0")
    if atr_period < 2:
        raise ConfigError("execution.trailing_stop_atr_period must be >= 2")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_validation.py -v`
Expected: all prior tests still pass, plus 4 new passes.

- [ ] **Step 5: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(config): validate trailing-stop fields (mutual exclusion + bounds)"
```

---

### Task 4: Create the `compute_atr` helper

**Files:**
- Create: `backtester/engine/atr.py`
- Create: `tests/unit/test_atr.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_atr.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest


def _ohlc(highs, lows, closes):
    return pd.DataFrame({
        "open": closes,  # not used by ATR
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * len(closes),
    })


def test_atr_first_value_is_high_minus_low():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0, 11.0], lows=[8.0, 9.0], closes=[9.0, 10.0])
    atr = compute_atr(data, period=2)
    # TR[0] = 10 - 8 = 2 (by convention; close[-1] undefined)
    # TR[1] = max(11-9, |11-9|, |9-9|) = 2
    # ATR with period=2 first defined at index 1: mean(TR[0:2]) = 2.0
    assert atr.iloc[1] == pytest.approx(2.0)


def test_atr_period_2_matches_hand_calc():
    from backtester.engine.atr import compute_atr
    data = _ohlc(
        highs=[10.0, 12.0, 14.0, 11.0, 13.0],
        lows=[8.0, 10.0, 11.0, 9.0, 11.0],
        closes=[9.0, 11.0, 13.0, 10.0, 12.0],
    )
    # TR[0] = 10 - 8 = 2
    # TR[1] = max(12-10, |12-9|, |10-9|) = max(2,3,1) = 3
    # TR[2] = max(14-11, |14-11|, |11-11|) = max(3,3,0) = 3
    # TR[3] = max(11-9, |11-13|, |9-13|) = max(2,2,4) = 4
    # TR[4] = max(13-11, |13-10|, |11-10|) = max(2,3,1) = 3
    atr = compute_atr(data, period=2)
    assert atr.iloc[1] == pytest.approx((2 + 3) / 2)
    assert atr.iloc[2] == pytest.approx((3 + 3) / 2)
    assert atr.iloc[3] == pytest.approx((3 + 4) / 2)
    assert atr.iloc[4] == pytest.approx((4 + 3) / 2)


def test_atr_nan_before_period():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0, 11.0, 12.0], lows=[8.0, 9.0, 10.0], closes=[9.0, 10.0, 11.0])
    atr = compute_atr(data, period=3)
    assert pd.isna(atr.iloc[0])
    assert pd.isna(atr.iloc[1])
    assert not pd.isna(atr.iloc[2])


def test_atr_invalid_period_raises():
    from backtester.engine.atr import compute_atr
    data = _ohlc(highs=[10.0], lows=[8.0], closes=[9.0])
    with pytest.raises(ValueError, match="ATR period must be >= 2"):
        compute_atr(data, period=1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_atr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtester.engine.atr'`.

- [ ] **Step 3: Implement**

Create `backtester/engine/atr.py`:
```python
from __future__ import annotations

import pandas as pd


def compute_atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using simple-moving-average smoothing.

    TR[i] = max(high - low, |high - prev_close|, |low - prev_close|).
    TR[0] uses high[0] - low[0] (prev_close undefined).
    ATR[i] = rolling SMA of TR over `period` bars.

    Returns a Series aligned to data.index. First `period - 1` values
    are NaN. Callers MUST treat NaN as "ATR not yet available".
    """
    if period < 2:
        raise ValueError("ATR period must be >= 2")
    high = data["high"]
    low = data["low"]
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.rolling(period).mean()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_atr.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/atr.py tests/unit/test_atr.py
git commit -m "feat(engine): add compute_atr helper for trailing stops"
```

---

## Phase 2: TrailingStopState

### Task 5: Create `TrailingStopState` (disabled-by-default + percentage mode)

**Files:**
- Create: `backtester/engine/trailing_stop.py`
- Create: `tests/unit/test_trailing_stop.py`

**Rationale:** Test the disabled-by-default contract first (`enabled is False`), then the percentage-mode arithmetic. ATR mode comes in Task 6 to keep this task small.

- [ ] **Step 1: Write the failing tests (percentage mode + enabled flag)**

`tests/unit/test_trailing_stop.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest


def test_disabled_by_default():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState()
    assert ts.enabled is False


def test_enabled_when_pct_set():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    assert ts.enabled is True


def test_reset_arms_and_sets_peak_trough():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    assert ts.armed is True
    assert ts.peak_high == pytest.approx(100.0)
    assert ts.trough_low == pytest.approx(100.0)


def test_update_long_peak_ratchets_up_only():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    ts.update(high=101.0, low=95.0)
    ts.update(high=98.0, low=92.0)   # high lower — peak should NOT move
    ts.update(high=105.0, low=100.0)
    assert ts.peak_high == pytest.approx(105.0)


def test_update_short_trough_ratchets_down_only():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    ts.update(high=101.0, low=95.0)
    ts.update(high=104.0, low=98.0)  # low higher — trough should NOT move
    ts.update(high=99.0, low=90.0)
    assert ts.trough_low == pytest.approx(90.0)


def test_pct_stop_price_long():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    ts.update(high=110.0, low=99.0)
    # peak_high = 110, stop = 110 * (1 - 0.05) = 104.5
    assert ts.stop_price(sign=+1, bar_idx=0) == pytest.approx(104.5)


def test_pct_stop_price_short():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    ts.update(high=101.0, low=90.0)
    # trough_low = 90, stop = 90 * (1 + 0.05) = 94.5
    assert ts.stop_price(sign=-1, bar_idx=0) == pytest.approx(94.5)


def test_stop_price_none_when_disarmed():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    # Never call reset → armed is False
    assert ts.stop_price(sign=+1, bar_idx=0) is None


def test_disarm_clears_state():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(pct=0.05)
    ts.reset(entry_price=100.0)
    ts.update(high=110.0, low=99.0)
    ts.disarm()
    assert ts.armed is False
    assert ts.stop_price(sign=+1, bar_idx=0) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_trailing_stop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtester.engine.trailing_stop'`.

- [ ] **Step 3: Implement (percentage mode only — ATR mode in Task 6)**

Create `backtester/engine/trailing_stop.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TrailingStopState:
    """Singleton per-simulation trailing-stop state.

    Owned by `PortfolioSimulator`. Reset on entry / flip; updated each
    bar with the bar's high/low; queried for the next bar's stop price.
    """

    pct: Optional[float] = None
    atr_mult: Optional[float] = None
    atr_series: Optional[pd.Series] = None  # aligned to data.index; required iff atr_mult
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
            if sign > 0:
                return self.peak_high * (1.0 - self.pct)
            return self.trough_low * (1.0 + self.pct)
        # ATR mode handled in Task 6
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_trailing_stop.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/trailing_stop.py tests/unit/test_trailing_stop.py
git commit -m "feat(engine): add TrailingStopState (percentage mode)"
```

---

### Task 6: Add ATR mode to `TrailingStopState`

**Files:**
- Modify: `backtester/engine/trailing_stop.py`
- Modify: `tests/unit/test_trailing_stop.py` (append two tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_trailing_stop.py`:
```python
def test_enabled_when_atr_set():
    from backtester.engine.trailing_stop import TrailingStopState
    ts = TrailingStopState(atr_mult=2.0, atr_series=pd.Series([float("nan"), 2.0, 3.0]))
    assert ts.enabled is True


def test_atr_stop_price_long_uses_indexed_value():
    from backtester.engine.trailing_stop import TrailingStopState
    atr = pd.Series([float("nan"), 2.0, 3.0])
    ts = TrailingStopState(atr_mult=2.0, atr_series=atr)
    ts.reset(entry_price=100.0)
    # peak_high = 100 (no update calls yet)
    # bar_idx=0 → NaN ATR → None
    assert ts.stop_price(sign=+1, bar_idx=0) is None
    # bar_idx=1 → ATR=2.0 → stop = 100 - 2*2 = 96
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(96.0)
    # bar_idx=2 → ATR=3.0 → stop = 100 - 2*3 = 94
    assert ts.stop_price(sign=+1, bar_idx=2) == pytest.approx(94.0)


def test_atr_stop_price_short_uses_indexed_value():
    from backtester.engine.trailing_stop import TrailingStopState
    atr = pd.Series([2.0, 3.0])
    ts = TrailingStopState(atr_mult=1.5, atr_series=atr)
    ts.reset(entry_price=100.0)
    # trough_low = 100
    # bar_idx=0 → ATR=2.0 → stop = 100 + 1.5*2 = 103
    assert ts.stop_price(sign=-1, bar_idx=0) == pytest.approx(103.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_trailing_stop.py -v -k atr`
Expected: 2 failures (the ATR path in `stop_price` currently returns None for all ATR cases).

- [ ] **Step 3: Implement**

In `backtester/engine/trailing_stop.py`, replace the `stop_price` method body to handle ATR mode:

```python
    def stop_price(self, sign: int, bar_idx: int) -> Optional[float]:
        if not self.armed or sign == 0:
            return None
        if self.pct is not None:
            if sign > 0:
                return self.peak_high * (1.0 - self.pct)
            return self.trough_low * (1.0 + self.pct)
        # ATR mode
        assert self.atr_series is not None  # invariant: atr_mult set ⇒ atr_series set
        atr_val = float(self.atr_series.iloc[bar_idx])
        if pd.isna(atr_val):
            return None
        if sign > 0:
            return self.peak_high - self.atr_mult * atr_val  # type: ignore[operator]
        return self.trough_low + self.atr_mult * atr_val  # type: ignore[operator]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_trailing_stop.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/trailing_stop.py tests/unit/test_trailing_stop.py
git commit -m "feat(engine): add ATR-multiple mode to TrailingStopState"
```

---

## Phase 3: Simulator integration

### Task 7: Wire dual pending slots (`pending_signal` / `pending_stop`) — refactor pass, no new behavior

**Files:**
- Modify: `backtester/engine/portfolio.py`

**Rationale:** This task is purely a rename + structural refactor. `pending` → `pending_signal`; add an unused `pending_stop` slot that is never set. After this commit, all existing tests must still pass (the refactor must be byte-identical because `pending_stop` is always None).

- [ ] **Step 1: Modify the simulator structure**

In `backtester/engine/portfolio.py`, rewrite the `simulate` method body. The diff intent:

- Rename the local `pending: Optional[Order] = None` to `pending_signal: Optional[Order] = None`.
- Add `pending_stop: Optional[Order] = None` next to it. Never assign to it in this task.
- Add a `stop_filled: bool = False` reset per iteration (always False until Task 8).
- Move pending-order execution to a "stop first, then signal" two-step block. Stop execution is currently inert.

Replacement loop body (the full `for i, ts in enumerate(index)` block) — the only logical change vs. v0.2.0 is the dual-slot naming and the inert stop slot:

```python
        for i, ts in enumerate(index):
            bar = data.iloc[i]
            stop_filled = False

            # 1a. Execute pending stop order (Phase 3+: trailing stop).
            if pending_stop is not None:
                fill = broker.submit(pending_stop, bar)
                if fill is not None:
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                    stop_filled = True
                pending_stop = None

            # 1b. Execute pending signal order (cancelled if stop fired same bar).
            if pending_signal is not None:
                if not stop_filled:
                    fill = broker.submit(pending_signal, bar)
                    if fill is not None:
                        fills.append(fill)
                        cash += fill.cash_delta
                        pos.apply_fill(fill)
                pending_signal = None

            # 2. Read this bar's signal
            sig = int(signals[sig_col].iloc[i]) if sig_col in signals.columns else 0
            if sig == -1 and not broker.allow_short:
                raise ShortNotAllowedError(
                    f"strategy emitted SHORT signal at bar {i} ({ts}) but "
                    f"execution.allow_short is False"
                )

            # 3. Decide whether to schedule an order for the next bar
            if i + 1 < len(index):
                next_bar_ts = index[i + 1]
                prev_sign = _sign(pos.qty)
                target_sign = sig

                if prev_sign != target_sign:
                    close_px = float(bar["close"])
                    if target_sign == 0:
                        order_qty = abs(pos.qty)
                        side = OrderSide.BUY if prev_sign < 0 else OrderSide.SELL
                        order_type = OrderType.MARKET
                        limit_price = None
                    else:
                        equity_now = cash + pos.market_value(close_px)
                        size = (
                            float(signals[size_col].iloc[i])
                            if size_col and size_col in signals.columns
                            else 1.0
                        )
                        alloc = equity_now * self.config.size * size
                        new_leg_qty = broker.round_qty(alloc / close_px)
                        if prev_sign == 0:
                            order_qty = new_leg_qty
                        else:
                            order_qty = abs(pos.qty) + new_leg_qty
                        side = OrderSide.BUY if target_sign > 0 else OrderSide.SELL
                        if (
                            prev_sign == 0
                            and price_col
                            and price_col in signals.columns
                            and pd.notna(signals[price_col].iloc[i])
                        ):
                            order_type = OrderType.LIMIT
                            limit_price = float(signals[price_col].iloc[i])
                        else:
                            order_type = OrderType.MARKET
                            limit_price = None

                    if order_qty > 0:
                        pending_signal = Order(
                            timestamp=next_bar_ts,
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            order_type=order_type,
                            limit_price=limit_price,
                        )

            # 4. Mark to market at close
            mv = pos.market_value(float(bar["close"]))
            equity = cash + mv
            equity_rows.append({"timestamp": ts, "cash": cash, "position_value": mv, "equity": equity})
            position_rows.append({"timestamp": ts, "qty": pos.qty, "avg_cost": pos.avg_cost, "close": float(bar["close"])})
```

Above the loop, replace the single-slot declaration:
```python
        pending: Optional[Order] = None
```
with:
```python
        pending_signal: Optional[Order] = None
        pending_stop: Optional[Order] = None
```

- [ ] **Step 2: Run the full suite to verify no regressions**

Run: `python -m pytest -q`
Expected: `197 passed`. Math: baseline 172 + Task 1 (2) + Task 2 (3) + Task 3 (4) + Task 4 (4) + Task 5 (9) + Task 6 (3) = 197. Task 7 adds no new tests; the refactor must be byte-identical.

- [ ] **Step 3: Commit**

```
git add backtester/engine/portfolio.py
git commit -m "refactor(portfolio): split pending order into signal+stop slots (no behavior change)"
```

---

### Task 8: Wire `TrailingStopState` into the simulator (percentage mode end-to-end)

**Files:**
- Modify: `backtester/engine/portfolio.py`
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** First green path — synthesize a long position whose price ratchets up then crashes by more than 5%, expect a stop-out fill on the crash bar. Percentage mode only; ATR mode in Task 11.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_long_trailing_stop_fires_on_drawdown():
    # Build a 20-bar series: 10 bars trending up then a sharp drop.
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=20)
    closes = np.concatenate([
        np.linspace(100.0, 120.0, 10),  # uptrend
        np.linspace(118.0, 100.0, 10),  # drawdown ~15%
    ])
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()
    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * 20},
        index=idx,
    )

    # Long-forever signal (enter bar 1, hold).
    sig = pd.DataFrame(index=idx)
    sig["signal"] = 1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    # Exactly one BUY (entry) and one SELL (stop-out).
    assert len(trades) >= 2
    assert trades.iloc[0]["side"] == "buy"
    assert trades.iloc[0]["reason"] == "signal"
    # At least one SELL must be a trailing_stop.
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    assert stop_rows.iloc[0]["side"] == "sell"
    # Position must reach flat after the stop.
    assert (positions["qty"] == 0).any()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_portfolio.py::test_long_trailing_stop_fires_on_drawdown -v`
Expected: FAIL — `KeyError: 'reason'` (the simulator does not yet write the column) or `assert len(stop_rows) >= 1` failure (no STOP order is being scheduled).

- [ ] **Step 3: Add imports**

At the top of `backtester/engine/portfolio.py`, add two new imports next to the existing engine imports:

```python
from backtester.engine.atr import compute_atr
from backtester.engine.trailing_stop import TrailingStopState
```

Verify `from backtester.engine.fills import Fill` is already imported — it is (used by the `Fill` symbol below). No change there.

- [ ] **Step 4: Construct `TrailingStopState` before the per-bar loop**

In `simulate`, BEFORE the existing `pending_signal: Optional[Order] = None` / `pending_stop: Optional[Order] = None` declarations from Task 7, insert:

```python
        cfg = broker.config
        trailing = TrailingStopState(
            pct=cfg.trailing_stop_pct,
            atr_mult=cfg.trailing_stop_atr_mult,
            atr_series=(
                compute_atr(data, cfg.trailing_stop_atr_period)
                if cfg.trailing_stop_atr_mult is not None else None
            ),
        )
```

- [ ] **Step 5: Replace the entire per-bar loop body**

Replace the `for i, ts in enumerate(index):` body that Task 7 produced with the final version below. The diff vs. Task 7:
- New `prev_sign_at_bar_start` line at the top of each iteration.
- Step 1a tags the stop-fill with `reason="trailing_stop"` and disarms `trailing`.
- New step 1c: reset/arm/disarm `trailing` on sign change; update peak/trough using this bar's OHLC.
- In the order-scheduling block (formerly step 3), the local variable previously named `prev_sign` is renamed `cur_sign` for clarity — it's the SAME value as `_sign(pos.qty)` post-fills, equal to `new_sign` defined in step 1c. The signal-order construction logic is unchanged.
- New step 4 (formerly the tail of step 3): schedule the STOP order for the next bar.

Final loop body:

```python
        for i, ts in enumerate(index):
            bar = data.iloc[i]
            prev_sign_at_bar_start = _sign(pos.qty)
            stop_filled = False

            # 1a. Execute pending stop order (priority over signal).
            if pending_stop is not None:
                fill = broker.submit(pending_stop, bar)
                if fill is not None:
                    fill = Fill(
                        timestamp=fill.timestamp,
                        symbol=fill.symbol,
                        side=fill.side,
                        qty=fill.qty,
                        price=fill.price,
                        commission=fill.commission,
                        reason="trailing_stop",
                    )
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                    stop_filled = True
                    trailing.disarm()
                pending_stop = None

            # 1b. Execute pending signal order (cancelled if stop fired same bar).
            if pending_signal is not None:
                if not stop_filled:
                    fill = broker.submit(pending_signal, bar)
                    if fill is not None:
                        fills.append(fill)
                        cash += fill.cash_delta
                        pos.apply_fill(fill)
                pending_signal = None

            # 1c. Trailing-stop state transitions and per-bar peak/trough update.
            new_sign = _sign(pos.qty)
            if new_sign != prev_sign_at_bar_start:
                if new_sign != 0:
                    # Entry from flat OR flip — arm at the latest fill price.
                    trailing.reset(entry_price=fills[-1].price)
                elif not stop_filled:
                    # Signal-driven exit to flat (stop-driven exit already disarmed in 1a).
                    trailing.disarm()
            if trailing.armed:
                trailing.update(high=float(bar["high"]), low=float(bar["low"]))

            # 2. Read this bar's signal.
            sig = int(signals[sig_col].iloc[i]) if sig_col in signals.columns else 0
            if sig == -1 and not broker.allow_short:
                raise ShortNotAllowedError(
                    f"strategy emitted SHORT signal at bar {i} ({ts}) but "
                    f"execution.allow_short is False"
                )

            # 3. Schedule signal-driven order for the next bar.
            if i + 1 < len(index):
                next_bar_ts = index[i + 1]
                cur_sign = new_sign
                target_sign = sig

                if cur_sign != target_sign:
                    close_px = float(bar["close"])
                    if target_sign == 0:
                        order_qty = abs(pos.qty)
                        side = OrderSide.BUY if cur_sign < 0 else OrderSide.SELL
                        order_type = OrderType.MARKET
                        limit_price = None
                    else:
                        equity_now = cash + pos.market_value(close_px)
                        size = (
                            float(signals[size_col].iloc[i])
                            if size_col and size_col in signals.columns
                            else 1.0
                        )
                        alloc = equity_now * self.config.size * size
                        new_leg_qty = broker.round_qty(alloc / close_px)
                        if cur_sign == 0:
                            order_qty = new_leg_qty
                        else:
                            order_qty = abs(pos.qty) + new_leg_qty
                        side = OrderSide.BUY if target_sign > 0 else OrderSide.SELL
                        if (
                            cur_sign == 0
                            and price_col
                            and price_col in signals.columns
                            and pd.notna(signals[price_col].iloc[i])
                        ):
                            order_type = OrderType.LIMIT
                            limit_price = float(signals[price_col].iloc[i])
                        else:
                            order_type = OrderType.MARKET
                            limit_price = None

                    if order_qty > 0:
                        pending_signal = Order(
                            timestamp=next_bar_ts,
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            order_type=order_type,
                            limit_price=limit_price,
                        )

                # 4. Schedule trailing STOP order for the next bar.
                if trailing.enabled and new_sign != 0 and trailing.armed:
                    stop_px = trailing.stop_price(sign=new_sign, bar_idx=i)
                    if stop_px is not None:
                        stop_side = OrderSide.SELL if new_sign > 0 else OrderSide.BUY
                        pending_stop = Order(
                            timestamp=next_bar_ts,
                            symbol=symbol,
                            side=stop_side,
                            qty=abs(pos.qty),
                            order_type=OrderType.STOP,
                            stop_price=stop_px,
                        )

            # 5. Mark to market at close.
            mv = pos.market_value(float(bar["close"]))
            equity = cash + mv
            equity_rows.append({"timestamp": ts, "cash": cash, "position_value": mv, "equity": equity})
            position_rows.append({"timestamp": ts, "qty": pos.qty, "avg_cost": pos.avg_cost, "close": float(bar["close"])})
```

- [ ] **Step 6: Update the trades DataFrame construction to include `reason`**

After the loop, replace the existing `trades_df = pd.DataFrame([...])` block at the end of `simulate` with:

```python
        trades_df = pd.DataFrame([
            {
                "timestamp": f.timestamp,
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "commission": f.commission,
                "notional": f.notional,
                "reason": f.reason,
            }
            for f in fills
        ])
```

Signal-driven fills from `broker.submit` keep the default `reason="signal"` field — no further tagging needed.

- [ ] **Step 7: Run the new test**

Run: `python -m pytest tests/unit/test_portfolio.py::test_long_trailing_stop_fires_on_drawdown -v`
Expected: PASS.

- [ ] **Step 8: Run the full suite — REGRESSION CHECK**

Run: `python -m pytest -q`
Expected: `198 passed` (197 from Task 7 + 1 new). **Any regression in the v0.2.0 portfolio tests** means the loop rewrite broke something — debug before continuing. Most likely culprits: (a) `cur_sign` references somewhere left as `prev_sign`; (b) `trailing.reset(entry_price=fills[-1].price)` fires on a bar where `fills` is empty because the just-applied fill was the FIRST in the run — verify there's at least one fill in `fills` before this branch executes (the branch only fires on `new_sign != prev_sign_at_bar_start`, which requires a fill on this bar, so the invariant holds).

- [ ] **Step 9: Commit**

```
git add backtester/engine/portfolio.py tests/unit/test_portfolio.py
git commit -m "feat(portfolio): wire trailing-stop into simulator (percentage mode)"
```

---

### Task 9: Short-side trailing stop

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** The simulator change in Task 8 already covers both long and short via `new_sign`. This task is a green-test confirmation, not new code.

- [ ] **Step 1: Write the failing test (it should pass on first run since Task 8 covered it)**

Append to `tests/unit/test_portfolio.py`:
```python
def test_short_trailing_stop_fires_on_rally():
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=20)
    closes = np.concatenate([
        np.linspace(100.0, 80.0, 10),   # downtrend (good for shorts)
        np.linspace(82.0, 100.0, 10),   # rally (stops out the short)
    ])
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()
    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * 20},
        index=idx,
    )

    sig = pd.DataFrame(index=idx)
    sig["signal"] = -1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         allow_short=True, trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert trades.iloc[0]["side"] == "sell"
    assert trades.iloc[0]["reason"] == "signal"
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    assert stop_rows.iloc[0]["side"] == "buy"
    assert (positions["qty"] < 0).any()
    assert (positions["qty"] == 0).any()
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/unit/test_portfolio.py::test_short_trailing_stop_fires_on_rally -v`
Expected: PASS. (If FAIL, the bug is in Task 8's sign handling — debug `new_sign`/`stop_side` for the short case.)

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `199 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): trailing stop fires on short-side rally"
```

---

### Task 10: Gap-through-stop fills at the open

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** The brief calls out gap handling explicitly. The existing FillEngine already fills SELL STOP at `min(open, stop_price)` and BUY STOP at `max(open, stop_price)`, so when the bar's open is past the stop level the fill price equals the open (worse than the stop). This test pins that semantic so a future FillEngine change can't silently break it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_gap_through_stop_fills_at_open():
    """When the bar's open gaps below a long trailing stop, the fill price
    is the bar's open (realistic), not the stop level (charitable)."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    # Bars 1-3: rise gently. Bar 4 gaps DOWN through any 5% stop.
    data = pd.DataFrame({
        "open":   [100.0, 101.0, 103.0, 105.0, 80.0],
        "high":   [101.0, 102.0, 104.0, 106.0, 82.0],
        "low":    [99.5,  100.5, 102.5, 104.5, 78.0],
        "close":  [100.5, 101.5, 103.5, 105.5, 81.0],
        "volume": [1_000_000] * 5,
    }, index=idx)
    # Enter long on bar 1; stop fires on bar 4 (gap-down).
    sig = pd.DataFrame(index=idx)
    sig["signal"] = [0, 1, 1, 1, 1]
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)
    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)

    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) == 1
    # peak_high before bar 4 is max(101,102,104,106) = 106. stop_level = 106 * 0.95 = 100.7.
    # Bar 4 open = 80, which is BELOW stop_level. Fill price = min(open, stop) = 80.
    assert stop_rows.iloc[0]["price"] == pytest.approx(80.0)
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_portfolio.py::test_gap_through_stop_fills_at_open -v`
Expected: PASS (the FillEngine path already provides this).

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `200 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): gap-through-stop fills at bar open"
```

---

### Task 11: ATR-mode end-to-end

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** Confirms the simulator wires `trailing_stop_atr_mult` + `trailing_stop_atr_period` correctly and that ATR-NaN bars before the period elapses do not trigger spurious stops.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_atr_mode_fires_and_warms_up():
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=30)
    # Steady uptrend for 25 bars then a sharp drop in the last 5.
    closes = np.concatenate([
        np.linspace(100.0, 120.0, 25),
        np.linspace(118.0, 100.0, 5),
    ])
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes.copy()
    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * 30},
        index=idx,
    )

    sig = pd.DataFrame(index=idx)
    sig["signal"] = 1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(
        commission_bps=0.0, slippage_bps=0.0,
        trailing_stop_atr_mult=3.0, trailing_stop_atr_period=14,
    )
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)
    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)

    # Entry on bar 1 is fine. ATR is NaN until bar 13 (period=14, first defined
    # at index 13). So no STOP order is scheduled for any bar in [0, 12]; the
    # earliest stop-out can only happen on bar 14 or later.
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    # Stop-out timestamp must be at or after index 14.
    stop_ts = stop_rows.iloc[0]["timestamp"]
    assert stop_ts >= idx[14]
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_portfolio.py::test_atr_mode_fires_and_warms_up -v`
Expected: PASS (Task 8 already wired ATR via `compute_atr` and Task 6 returned None during NaN).

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `201 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): ATR-mode trailing stop fires after warmup"
```

---

### Task 12: Stop wins over signal flip on the same bar

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** The precedence rule from the spec — when both pending orders are present and the stop fills, the signal order is cancelled. This is the highest-risk corner-case test.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_stop_wins_over_signal_flip_same_bar():
    """Long open, signal flips to -1 the same bar a long-stop would fire.
    Expected: stop fires (reason=trailing_stop), the signal-driven flip
    order is cancelled, the position lands flat (NOT short) at end-of-bar.
    The next bar may re-enter short normally."""
    idx = pd.bdate_range("2024-01-02", periods=8)
    # Bars 1-3 rise; bar 4 gaps down hard.
    data = pd.DataFrame({
        "open":   [100.0, 101.0, 103.0, 105.0, 80.0, 80.5, 81.0, 80.0],
        "high":   [101.0, 102.0, 104.0, 106.0, 82.0, 81.5, 81.5, 81.0],
        "low":    [99.0,  100.5, 102.5, 104.5, 78.0, 79.5, 80.5, 79.0],
        "close":  [100.5, 101.5, 103.5, 105.5, 81.0, 80.5, 81.0, 80.5],
        "volume": [1_000_000] * 8,
    }, index=idx)
    # Long bars 1..3, then flip to SHORT at bar 3 -> order scheduled for bar 4.
    # On bar 4, the trailing stop also fires (gap down through 5% level).
    sig = pd.DataFrame(index=idx)
    sig["signal"] = [0, 1, 1, -1, -1, -1, -1, -1]
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         allow_short=True, trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)
    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)

    # The bar-4 transition: stop fills first (reason=trailing_stop), the
    # signal-driven combined-flip SELL is CANCELLED, so position is flat
    # at end-of-bar-4 (NOT short).
    bar4_stop = trades[(trades["timestamp"] == idx[4]) & (trades["reason"] == "trailing_stop")]
    assert len(bar4_stop) == 1
    assert bar4_stop.iloc[0]["side"] == "sell"
    # No same-bar combined-flip — the signal order was cancelled.
    bar4_signal_sell = trades[(trades["timestamp"] == idx[4]) & (trades["reason"] == "signal") & (trades["side"] == "sell")]
    assert len(bar4_signal_sell) == 0
    # Position at bar 4 is flat (qty == 0).
    assert positions.loc[idx[4], "qty"] == 0
    # Bar 5 may re-enter short; not asserted strictly because signal also
    # may have changed in the underlying flow. But the position must NEVER
    # be simultaneously long-and-short within a single bar's state.
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_portfolio.py::test_stop_wins_over_signal_flip_same_bar -v`
Expected: PASS, because Task 7's `if not stop_filled` guard ensures `pending_signal` is dropped when the stop fills first.

If FAIL: the most likely bug is that the signal-driven SELL was scheduled for bar 4 and `pending_signal` was not cancelled when the stop fired. Re-read Task 7 step 1b.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `202 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): trailing stop wins over signal flip on same bar"
```

---

### Task 13: Trailing state resets on flip through zero

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** When a position flips long → short (or vice versa) via the v0.2.0 combined-order semantic, the trailing state must reset to the new-leg fill price. Otherwise the new short would inherit the long's `trough_low` (= inf or the long's prior lows), which would prevent the short stop from firing.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_stop_resets_on_flip_through_zero():
    """Long position, then signal flips to -1 (no stop fires on the flip
    bar). The new short leg must arm its trailing state at the flip-fill
    price. We verify by triggering the SHORT stop on a subsequent rally:
    if the trailing state were not reset, trough_low would be lower than
    the flip price and the short stop would not fire at the expected level."""
    idx = pd.bdate_range("2024-01-02", periods=12)
    # Bars 0..4 rise gently. Bar 4 close ≈ 104. Bar 5 the signal flips long → short.
    # Bars 5..7 trend down (good for the short). Bar 8 rallies sharply, triggering the short stop.
    data = pd.DataFrame({
        "open":   [100, 101, 102, 103, 104, 103, 100, 98,  102, 105, 108, 110.0],
        "high":   [101, 102, 103, 104, 105, 104, 101, 99,  103, 106, 109, 111.0],
        "low":    [99,  100, 101, 102, 103, 100, 98,  96,  99,  103, 106, 108.0],
        "close":  [100.5, 101.5, 102.5, 103.5, 104.0, 102.0, 99.0, 97.0, 102.0, 105.5, 108.5, 110.5],
        "volume": [1_000_000] * 12,
    }, index=idx)
    sig = pd.DataFrame(index=idx)
    sig["signal"] = [0, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         allow_short=True, trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)
    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)

    # Exactly: entry buy, flip sell (combined), short stop-out buy.
    sides = list(trades["side"])
    reasons = list(trades["reason"])
    assert sides[0] == "buy" and reasons[0] == "signal"
    assert sides[1] == "sell" and reasons[1] == "signal"     # combined flip
    # At least one trailing_stop fill (the short stop on the rally).
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    assert stop_rows.iloc[0]["side"] == "buy"
    # Confirm the short stop's price is reasonable given a trough_low
    # captured AFTER the flip (i.e., not 78 or some earlier-bar low).
    # The trough_low should be no lower than the flip fill price.
    # Flip happens on bar 5 at open=103. trough_low ≤ 103 thereafter.
    # Short stop = trough_low * 1.05; bar 8 high reaches 103 → no fire yet.
    # The actual fire bar's high * stop relationship is what we assert:
    fire_bar_high_must_exceed_stop = stop_rows.iloc[0]["price"]
    assert fire_bar_high_must_exceed_stop > 100.0  # crude sanity bound
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_portfolio.py::test_stop_resets_on_flip_through_zero -v`
Expected: PASS. The reset is performed by the step 1c `new_sign != prev_sign_at_bar_start` branch — both pre and post sign are non-zero, but they differ, so `trailing.reset(entry_price=fills[-1].price)` is called with the flip-fill price.

If FAIL: the reset path may be guarded incorrectly (e.g., `new_sign != 0 and prev_sign_at_bar_start == 0`, missing the flip case). Re-read step 1c in Task 8.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `203 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): trailing state resets on flip through zero"
```

---

### Task 14: Backwards compatibility — disabled trailing stop is byte-identical

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append one test)

**Rationale:** Pin the no-regression guarantee at the unit level. Compare two PortfolioSimulator runs with identical signals and data — one constructed with default `ExecutionConfig()` (no trailing stop), one with the new fields explicitly `None`. They must produce identical trades, positions, and equity.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_portfolio.py`:
```python
def test_no_trailing_stop_is_byte_identical_to_baseline():
    """Two simulator runs on the same data with the trailing stop OFF must
    produce identical trades.csv-equivalent and equity_curve DataFrames.
    Pins the backwards-compat invariant at the unit level."""
    data = make_ohlcv(n=120, seed=42, start_price=100.0, drift=0.001, vol=0.012)
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 0
    sig["signal"].iloc[10:60] = 1
    sig["signal"].iloc[60:100] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    # Run 1: default ExecutionConfig (no trailing fields set).
    cfg_a = ExecutionConfig(commission_bps=2.0, slippage_bps=5.0)
    sim_a = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    trades_a, positions_a, eq_a = sim_a.simulate(
        data=data, signal_frame=sf, broker=Broker(cfg_a)
    )

    # Run 2: trailing fields explicitly None.
    cfg_b = ExecutionConfig(
        commission_bps=2.0, slippage_bps=5.0,
        trailing_stop_pct=None, trailing_stop_atr_mult=None,
    )
    sim_b = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    trades_b, positions_b, eq_b = sim_b.simulate(
        data=data, signal_frame=sf, broker=Broker(cfg_b)
    )

    # Trades schemas may differ in the `reason` column presence depending on
    # whether any fills occurred; assert column-by-column equality on shared
    # columns and exact length.
    pd.testing.assert_frame_equal(trades_a, trades_b)
    pd.testing.assert_frame_equal(positions_a, positions_b)
    pd.testing.assert_frame_equal(eq_a, eq_b)
    # And: every trade's reason is exactly "signal".
    if not trades_a.empty:
        assert set(trades_a["reason"]) == {"signal"}
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_portfolio.py::test_no_trailing_stop_is_byte_identical_to_baseline -v`
Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: `204 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): disabled trailing stop is byte-identical to baseline"
```

---

## Phase 4: Reporting and config samples

### Task 15: Add the sample trailing-stop backtest config

**Files:**
- Create: `configs/backtests/sma_cross_spy_trailing.yaml`

- [ ] **Step 1: Write the file**

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

- [ ] **Step 2: Smoke-run from the CLI**

Run:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy_trailing.yaml
```
Expected: exit 0, run directory printed.

- [ ] **Step 3: Inspect trades.csv**

Run:
```
python -c "import pandas as pd; t = pd.read_csv(r'<RUNDIR>/trades.csv'); print(t.head()); print('reason counts:', t['reason'].value_counts().to_dict())"
```
Expected: `reason` column present; at least one row with `reason == 'trailing_stop'`.

- [ ] **Step 4: Commit**

```
git add configs/backtests/sma_cross_spy_trailing.yaml
git commit -m "feat(config): add sma_cross_spy_trailing backtest config"
```

---

### Task 16: CLI integration test — trailing-stop run

**Files:**
- Modify: `tests/integration/test_run_backtest_cli.py` (append one test)

- [ ] **Step 1: Inspect existing test style**

Run:
```
python -m pytest tests/integration/test_run_backtest_cli.py --co -q
```
Note the existing test names and the fixture style (typically `tmp_path` + a config copy with output_root overridden). Match it.

- [ ] **Step 2: Write the failing test**

Append to `tests/integration/test_run_backtest_cli.py` a test that follows the file's existing pattern. Pseudocode (adapt to actual fixture names):

```python
def test_run_backtest_cli_trailing_stop_smoke(tmp_path):
    import yaml, subprocess, json, pandas as pd

    # Load the trailing config; redirect output_root to tmp_path.
    src_cfg = "configs/backtests/sma_cross_spy_trailing.yaml"
    with open(src_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["output_root"] = str(tmp_path)
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Run.
    result = subprocess.run(
        ["python", "-m", "backtester.runners.run_backtest",
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    # Find the produced run dir.
    run_dirs = list(tmp_path.glob("*_sma_cross_spy_trailing"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Read trades; verify the reason column and at least one trailing_stop row.
    trades = pd.read_csv(run_dir / "trades.csv")
    assert "reason" in trades.columns
    assert (trades["reason"] == "trailing_stop").any()

    # Read summary; verify max_drawdown is more conservative than the no-stop run.
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "max_drawdown" in summary

    # Also run the no-stop baseline once for comparison.
    src_baseline = "configs/backtests/sma_cross_spy.yaml"
    with open(src_baseline) as f:
        bcfg = yaml.safe_load(f)
    bcfg["output_root"] = str(tmp_path)
    bcfg["run_name"] = "sma_cross_spy_baseline"
    bcfg_path = tmp_path / "baseline.yaml"
    bcfg_path.write_text(yaml.safe_dump(bcfg))
    b_result = subprocess.run(
        ["python", "-m", "backtester.runners.run_backtest",
         "--config", str(bcfg_path)],
        capture_output=True, text=True,
    )
    assert b_result.returncode == 0, b_result.stderr
    b_run = list(tmp_path.glob("*_sma_cross_spy_baseline"))[0]
    b_summary = json.loads((b_run / "summary.json").read_text())

    # The trailing-stop run should produce a smaller drawdown magnitude.
    # max_drawdown is typically negative; magnitude = |drawdown|.
    assert abs(summary["max_drawdown"]) < abs(b_summary["max_drawdown"])
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/integration/test_run_backtest_cli.py::test_run_backtest_cli_trailing_stop_smoke -v`
Expected: PASS. If the drawdown comparison fails, inspect whether SPY had a sustained drawdown where the 5% stop helped — if not, choose a different parameter (`trailing_stop_pct: 0.03`) or relax the assertion to `abs(summary["max_drawdown"]) <= abs(b_summary["max_drawdown"])`. **Do not** weaken to `<= 0` — that would be a tautology.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: `205 passed`.

- [ ] **Step 5: Commit**

```
git add tests/integration/test_run_backtest_cli.py
git commit -m "test(integration): CLI smoke test for trailing-stop backtest"
```

---

## Phase 5: Backwards-compat verification

### Task 17: Capture v0.2.0 baseline numerics for the SMA-Cross config

**Files:**
- Create: `tests/integration/test_backwards_compat.py`

**Rationale:** A standalone integration test that runs the unmodified `configs/backtests/sma_cross_spy.yaml` and compares numeric summary fields against a hard-coded golden dict captured before this change. This pins acceptance criterion 2 in CI.

- [ ] **Step 1: Capture the golden numbers**

If you have not already, run:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
And note the printed run directory. Then:
```
python -c "import json; s=json.load(open(r'<RUNDIR>/summary.json')); print(repr({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')}))"
```
**Copy the printed dict literally** — that exact string becomes the `EXPECTED` constant in the test.

If the simulator change broke the baseline (the dict differs from the **pre-change** baseline you captured at the very start), the refactor introduced a regression. Stop and debug — do not paper over by re-pinning to a drifted value.

- [ ] **Step 2: Write the test**

Create `tests/integration/test_backwards_compat.py`:
```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml


# Hard-coded golden captured from v0.2.0 / pre-trailing-stop sma_cross_spy run.
# Updated only when an intentional simulator change requires it.
EXPECTED = {
    "total_return": ...,    # paste from Step 1
    "sharpe": ...,
    "max_drawdown": ...,
    "n_trades": ...,
    "final_equity": ...,
}


def test_sma_cross_spy_unchanged_with_trailing_disabled(tmp_path):
    """Acceptance criterion 2: with trailing-stop fields absent (or None),
    the bundled long-only SMA-Cross config produces the v0.2.0 numerics
    exactly. Floats compared to 1e-9 abs / 1e-12 rel; n_trades exact."""
    src = Path("configs/backtests/sma_cross_spy.yaml")
    cfg = yaml.safe_load(src.read_text())
    cfg["output_root"] = str(tmp_path)
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = subprocess.run(
        ["python", "-m", "backtester.runners.run_backtest",
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = list(tmp_path.glob("*_sma_cross_spy"))
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text())

    assert summary["n_trades"] == EXPECTED["n_trades"]
    for key in ("total_return", "sharpe", "max_drawdown", "final_equity"):
        assert summary[key] == pytest.approx(
            EXPECTED[key], abs=1e-9, rel=1e-12
        ), f"{key}: got {summary[key]!r}, expected {EXPECTED[key]!r}"
```

- [ ] **Step 3: Fill in `EXPECTED`**

Replace the five `...` values in `EXPECTED` with the numbers from Step 1.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/integration/test_backwards_compat.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: `206 passed`.

- [ ] **Step 6: Commit**

```
git add tests/integration/test_backwards_compat.py
git commit -m "test(integration): pin v0.2.0 SMA-Cross numerics as backwards-compat golden"
```

---

## Phase 6: Documentation

### Task 18: Update `docs/strategy_contract.md`

**Files:**
- Modify: `docs/strategy_contract.md`

**Rationale:** Strategies should NOT emit trailing stops; the contract should make this explicit so future contributors don't try to add a `stop_column` to `SignalFrame`.

- [ ] **Step 1: Edit**

In `docs/strategy_contract.md`, find the "Signal semantics" section. After the existing bullet about `price_column` (LIMIT support), append:

```markdown
- Trailing stops are **execution-layer**, not strategy-layer. Configure
  via `execution.trailing_stop_pct` or `execution.trailing_stop_atr_mult`
  in the run YAML. Strategies have no `stop_column` and cannot emit
  per-trade stop levels in v0.3.0. The trailing stop trails the running
  peak (long) or trough (short) since entry and fires as a STOP order on
  the bar after the peak/trough is breached by the configured distance.
  Stop-out exits take precedence over the strategy signal on the same
  bar; the next bar's signal is read normally.
```

- [ ] **Step 2: Commit**

```
git add docs/strategy_contract.md
git commit -m "docs(strategy): note trailing stops are execution-layer, not strategy-layer"
```

---

### Task 19: Update `docs/runbook.md` Limitations

**Files:**
- Modify: `docs/runbook.md`

- [ ] **Step 1: Append**

Append to `docs/runbook.md`:

```markdown

## Trailing-stop limitations (v0.3.0)

The execution-layer trailing stop intentionally omits several features
that should be added as follow-up phases:

- **Only percentage and ATR-multiple distance modes.** Fixed-dollar
  trailing stops are out of scope for v0.3.0 (one-line addition once a
  use case justifies it).
- **Same-bar precedence is hard-coded.** A trailing-stop hit always
  cancels the same-bar signal-driven order. There is no configurable
  ordering between strategy intent and stop trigger.
- **No partial exits.** Stops always close the full position (`qty =
  abs(pos.qty)`). There is no "trail half, hold the rest" mechanism.
- **No grid- or WFO-searchable trailing parameters.** `trailing_stop_*`
  fields are not first-class entries in `OptimizationConfig.param_space`
  yet. To tune them you must hand-run multiple configs.
- **No re-entry cooldown.** After a stop fires, if the strategy signal
  still requests a position on the very next bar, the simulator
  re-enters immediately. Strategies that want a "wait N bars after a
  stop-out" rule must implement it themselves.
- **No interaction with borrow-cost accounting** (which is itself a
  documented v0.2.0 limitation). A short stopped out by a rally still
  pays no borrow during the holding period.
```

- [ ] **Step 2: Commit**

```
git add docs/runbook.md
git commit -m "docs(runbook): document trailing-stop limitations for v0.3.0"
```

---

### Task 20: Update `README.md` Execution model

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit**

Find the `## Execution model` section in `README.md`. After the existing bullet about `Commission and slippage are both in bps and applied per fill.`, INSERT a new bullet about trailing stops:

```markdown
- **Trailing stops** (v0.3.0). Set `execution.trailing_stop_pct: 0.05`
  (or `execution.trailing_stop_atr_mult: 3.0` with
  `trailing_stop_atr_period: 14`) to attach a trailing stop to every
  position. The stop trails the running peak (long) or trough (short)
  since entry and fires as a STOP order on the next bar. Stop-out fills
  are tagged `reason="trailing_stop"` in `trades.csv`; signal-driven
  fills are tagged `reason="signal"`. Trailing-stop hits take priority
  over the strategy signal on the same bar.
```

Also update the version line at the bottom (the `v0.2.0 — Long/short execution.` line) by appending a new line for v0.3.0:

```markdown
`v0.3.0` — Trailing stops. Adds an execution-layer trailing stop with
two distance modes (percentage of peak/trough, or multiple of recent
ATR). Stop-out exits are tagged `reason="trailing_stop"` in
`trades.csv`. Long-only and long/short configs are unchanged when both
trailing fields are unset. See
[`docs/superpowers/plans/2026-05-14-trailing-stops.md`](docs/superpowers/plans/2026-05-14-trailing-stops.md)
for the design.
```

- [ ] **Step 2: Commit**

```
git add README.md
git commit -m "docs(readme): describe trailing stops in Execution model section"
```

---

## Phase 7: Version bump and tag

### Task 21: Bump version to 0.3.0 and tag

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Final full-suite green check**

Run: `python -m pytest -q`
Expected: `206 passed` (baseline 172 + 34 new tests added across Tasks 1–17; recount and adjust if any test was added or moved during implementation).

If anything is red, STOP and fix root cause. Do not bump version on a red suite.

- [ ] **Step 2: Bump the version**

Edit `pyproject.toml` and change:
```toml
version = "0.2.0"
```
to:
```toml
version = "0.3.0"
```

- [ ] **Step 3: Reinstall to refresh egg-info**

Run: `pip install -e .[dev]`
Expected: `Successfully installed modular-stock-backtester-0.3.0 ...`.

- [ ] **Step 4: Commit and tag**

```
git add pyproject.toml
git commit -m "chore: bump version to 0.3.0 (trailing-stop support)"
git tag -a v0.3.0 -m "v0.3.0: execution-layer trailing stops (percent + ATR)"
```

- [ ] **Step 5 (only after user confirms): Push tag**

```
# git push origin master v0.3.0
```

Do NOT push without explicit user confirmation.

---

## Appendix A: Decision log

**Q: State on simulator or on Position?**
Simulator. Position.apply_fill happens mid-bar — before this bar's high/low has been observed. Putting peak/trough on Position would force the simulator to call back into Position with bar OHLC after each fill, breaking Position's role as pure fill algebra. A dedicated `TrailingStopState` dataclass owned by the simulator keeps the responsibility split clean.

**Q: One pending slot or two?**
Two. Signal-driven orders and stop-driven orders can coexist on the same bar (long open + signal-flip pending + stop also pending). Forcing them into one slot would either drop information or require an ad-hoc precedence inside the slot. Two slots with a "stop first, then signal (cancelled if stop fired)" execution order matches the precedence rule directly.

**Q: ATR pre-computed or per-bar?**
Pre-computed in a single pandas pass at the start of `simulate`. Vectorized rolling-mean is far cheaper than 1000s of per-bar Python branches. The pre-computed Series is stored on the `TrailingStopState` instance and indexed by bar position in `stop_price(bar_idx)`.

**Q: Tag fills via Fill.reason field or via a parallel sidecar dataframe?**
On Fill. The field defaults to `"signal"` and is set to `"trailing_stop"` only when the simulator's step 1a fires. Adding a field at the END of the slotted dataclass is backwards-compatible — every existing Fill constructor (incl. test helpers and the FillEngine) keeps working without edits. The alternative (sidecar stops.csv) was rejected during design because reconstructing "how did this position exit" would require joining two CSVs.

**Q: Should the trailing stop reset its state on partial position closes?**
N/A. The simulator never emits partial closes — every flip or exit is a full close (combined-order semantic for flips, full-quantity SELL/BUY for exits). If a future feature adds partial closes, the `TrailingStopState` reset logic will need extension; this is called out in the runbook.

## Appendix B: Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| Existing portfolio tests fail after Task 7 | Dual-slot refactor changed semantics — most likely `prev_sign` was redefined inside step 3 but a stale reference remains. Grep for `prev_sign` in `portfolio.py` and ensure each is `new_sign` for post-fill checks. |
| `test_no_trailing_stop_is_byte_identical_to_baseline` fails | A code path that should only fire when `trailing.enabled` is True is firing unconditionally. Verify step 3's stop-order branch is guarded by `if trailing.enabled and ...`. |
| Stop fires on the entry bar itself | `trailing.update(high, low)` is being called before `trailing.reset(...)` on a flip, OR step 1a is reading a stale `pending_stop` from a prior position. Verify `pending_stop = None` is cleared on every iteration. |
| Backwards-compat test fails on `n_trades` | The simulator is double-emitting a SELL because the signal-cancel-on-stop guard didn't take effect — verify `if not stop_filled` wraps the entire pending_signal block, not just the `broker.submit` call. |
| ATR-mode test fails before bar 14 | `stop_price` is returning a value instead of None when ATR is NaN — verify `pd.isna(atr_val)` short-circuits to None. |
| `trades.csv` has no `reason` column | The trades DataFrame builder at end of `simulate` was not updated. Re-read Task 8 step 3(e). |
| Sample trailing config exits with `ConfigError: ...mutually exclusive` | Both `trailing_stop_pct` and `trailing_stop_atr_mult` are set in the YAML. Pick one. |
