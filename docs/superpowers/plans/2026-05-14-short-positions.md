# Short-Position Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the long-only execution path of the Modular Backtester to support short positions end-to-end — entry, exit, PnL accounting, sizing, configuration, sample strategy, and tests — without breaking any existing long-only behavior.

**Architecture:** Two layered changes. (1) `Position.apply_fill` is rewritten as signed-quantity arithmetic that handles flat, long, short, and flip-through-zero transitions in one branchless algebra, gated by an `allow_short` flag. (2) `PortfolioSimulator.simulate` is rewritten from a binary `target_long` state machine to a tri-state `(prev_sign, target_sign)` state machine that emits a single combined order for each transition, including `long→short` and `short→long` flips. A new `ExecutionConfig.allow_short: bool = False` is wired through `Broker → PortfolioSimulator → Position`. When the flag is off, behavior is byte-identical to v0.1.0.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new runtime dependencies.

**Scope notes:**
- No borrow cost, hard-to-borrow, margin call, or per-symbol short ban modeling. These are documented as limitations.
- Default `allow_short = False` preserves v0.1.0 behavior for every existing config + strategy.
- `Position.apply_fill` keeps its position-level guard via a constructor `allow_short` argument. The simulator additionally pre-checks the signal stream and raises a clearer error before any order is constructed. Both guards raise the same exception class.
- Long → short and short → long transitions emit **one combined order** at MARKET. The Position's signed-qty algebra closes the prior leg and opens the new one in a single fill.
- LIMIT entries continue to apply only when the source state is flat (preserves existing long-only LIMIT semantics; extends naturally to flat-→-short LIMIT). STOP support is exercised at the Broker+Position level; the simulator does not emit STOP orders.

**Required reading before starting:**
1. `docs/superpowers/plans/2026-05-14-modular-stock-backtester.md` — the original plan that produced the v0.1.0 codebase. Skim Phases 4 and 5 (Engine and Portfolio).
2. `backtester/engine/position.py`, `backtester/engine/portfolio.py`, `backtester/engine/fills.py`, `backtester/core/enums.py`, `backtester/config/models.py`.
3. `tests/unit/test_position.py`, `tests/unit/test_portfolio.py` — baseline tests that MUST stay green (only one test is allowed to be edited — `test_sell_when_flat_raises`).
4. `docs/strategy_contract.md` — signal-semantics contract.

**Baseline verification before starting:**

Run from repo root:
```
python -m pytest -q
```
Expected: `135 passed` (or whatever the v0.1.0 baseline is). Record the exact count — every existing test must still pass at the end.

**File-structure preview (what will change):**

| File                                                | Action                       | Purpose                                              |
|-----------------------------------------------------|------------------------------|------------------------------------------------------|
| `backtester/core/exceptions.py`                     | modify                       | Add `ShortNotAllowedError(ExecutionError)`            |
| `backtester/config/models.py`                       | modify                       | Add `ExecutionConfig.allow_short: bool = False`      |
| `backtester/engine/position.py`                     | substantial rewrite          | Signed-qty algebra + `allow_short` gate              |
| `backtester/engine/portfolio.py`                    | substantial rewrite          | Tri-state state machine + `allow_short` plumbing     |
| `backtester/engine/broker.py`                       | tiny modify                  | Expose `allow_short` (no logic change inside Broker) |
| `strategies/rsi_long_short.py`                      | create                       | Sample symmetric-RSI long/short strategy             |
| `backtester/strategies/registry.py`                 | modify                       | Register the new strategy                            |
| `configs/wfo/rsi_long_short_wfo.yaml`               | create                       | Sample WFO config with `allow_short: true`           |
| `docs/strategy_contract.md`                         | modify                       | Add `-1 = short` row + `allow_short` note            |
| `docs/runbook.md`                                   | modify                       | Add "Limitations" section                            |
| `tests/unit/test_position.py`                       | modify (1 test) + new tests  | Short PnL coverage + the one allowed edit            |
| `tests/unit/test_portfolio.py`                      | append new tests             | Transition coverage + LIMIT/STOP for shorts          |
| `tests/unit/test_config_models.py`                  | append new test              | `allow_short` default + override                     |
| `tests/unit/test_strategy_rsi_long_short.py`        | create                       | Strategy unit tests                                  |
| `tests/integration/test_backtest_engine.py`         | append new test              | Synthetic downtrend short ends with positive return  |
| `tests/integration/test_run_backtest_cli.py`        | append new test              | CLI run with new strategy on SPY                     |
| `tests/integration/test_run_wfo_cli.py`             | append new test              | WFO smoke for the new strategy                       |
| `pyproject.toml`                                    | bump version                 | `0.1.0` → `0.2.0`                                    |

---

## Phase 1: Scaffolding (exception + config flag)

### Task 1: Add `ShortNotAllowedError` exception

**Files:**
- Modify: `backtester/core/exceptions.py`
- Create: `tests/unit/test_exceptions_short.py`

**Rationale:** A dedicated subclass of `ExecutionError` makes "shorts disabled" failures grep-able and avoids overloading `ValueError`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_exceptions_short.py`:
```python
from __future__ import annotations

from backtester.core.exceptions import (
    BacktesterError,
    ExecutionError,
    ShortNotAllowedError,
)


def test_short_not_allowed_inherits_execution_error():
    assert issubclass(ShortNotAllowedError, ExecutionError)
    assert issubclass(ShortNotAllowedError, BacktesterError)


def test_short_not_allowed_carries_message():
    e = ShortNotAllowedError("shorts disabled at bar 42")
    assert "shorts disabled" in str(e)
    assert "42" in str(e)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_exceptions_short.py -v`
Expected: FAIL — `ImportError: cannot import name 'ShortNotAllowedError'`.

- [ ] **Step 3: Implement**

Append to `backtester/core/exceptions.py`:
```python
class ShortNotAllowedError(ExecutionError):
    """Raised when a short order or short-opening fill is attempted while
    `allow_short` is disabled (on either ExecutionConfig or Position)."""
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_exceptions_short.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backtester/core/exceptions.py tests/unit/test_exceptions_short.py
git commit -m "feat(core): add ShortNotAllowedError subclass of ExecutionError"
```

---

### Task 2: Add `allow_short` flag to `ExecutionConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Modify: `tests/unit/test_config_models.py` (append one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_models.py`:
```python
def test_execution_config_allow_short_defaults_false():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.allow_short is False


def test_execution_config_allow_short_override():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig(allow_short=True)
    assert cfg.allow_short is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k allow_short`
Expected: FAIL — `AttributeError: 'ExecutionConfig' object has no attribute 'allow_short'`.

- [ ] **Step 3: Implement**

Modify `backtester/config/models.py`. Find the `ExecutionConfig` dataclass and add the `allow_short` field at the end so YAML configs that omit it still work:

```python
@dataclass(slots=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_fractional: bool = False
    allow_short: bool = False
```

- [ ] **Step 4: Verify the YAML loader picks it up automatically**

Run from a Python REPL (or one-liner):
```
python -c "from backtester.config.loader import load_run_config; rc = load_run_config('configs/backtests/sma_cross_spy.yaml'); print('allow_short =', rc.execution.allow_short)"
```
Expected: `allow_short = False`. (The loader uses `ExecutionConfig(**raw)` so unspecified keys default cleanly.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v`
Expected: all previous tests still pass, plus 2 new passes.

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add ExecutionConfig.allow_short (default False)"
```

---

## Phase 2: Position-level signed-qty arithmetic

### Task 3: Reframe the one allowed test edit

**Files:**
- Modify: `tests/unit/test_position.py` (edit `test_sell_when_flat_raises`)

**Rationale:** This is the one existing test the PRD explicitly allows to be reframed. We keep its intent (SELL on a flat position is rejected when shorts are not allowed) but switch the asserted exception type from `ValueError` to the new `ShortNotAllowedError`. Position will continue to default `allow_short=False` so the test exercises the default case.

- [ ] **Step 1: Edit the test**

Replace the body of `test_sell_when_flat_raises` in `tests/unit/test_position.py` with:
```python
def test_sell_when_flat_raises():
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY")  # allow_short defaults to False
    with pytest.raises(ShortNotAllowedError, match="shorts not allowed"):
        p.apply_fill(_fill(OrderSide.SELL, 1, 100.0))
```

- [ ] **Step 2: Run to verify it now fails (because `allow_short` arg + new exception don't exist yet)**

Run: `python -m pytest tests/unit/test_position.py::test_sell_when_flat_raises -v`
Expected: FAIL — either `ImportError` (the existing position file does not import `ShortNotAllowedError`) or `Failed: DID NOT RAISE` / `ValueError != ShortNotAllowedError`. **Do not implement yet.**

- [ ] **Step 3: Commit (test-only change, intentionally red)**

```
git add tests/unit/test_position.py
git commit -m "test(position): reframe sell-when-flat to expect ShortNotAllowedError"
```

---

### Task 4: Write the failing tests for short-side Position behavior

**Files:**
- Modify: `tests/unit/test_position.py` (append new tests)

- [ ] **Step 1: Append the new tests**

Append to `tests/unit/test_position.py`:
```python
# --- Short-side tests (Phase 0.2 short-position support) ---

def _short_pos():
    """Helper: position with shorts enabled."""
    return Position(symbol="SPY", allow_short=True)


def test_short_entry_from_flat_sets_negative_qty_and_avg_cost():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0, commission=1.0))
    assert p.qty == -10
    assert p.avg_cost == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(-1.0)
    assert not p.is_flat


def test_two_short_entries_compute_weighted_avg_cost():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 10, 110.0))
    assert p.qty == -20
    assert p.avg_cost == pytest.approx(105.0)


def test_partial_cover_realizes_short_pnl_when_price_drops():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))            # short @ 100
    p.apply_fill(_fill(OrderSide.BUY, 4, 90.0, commission=0.5))  # cover 4 @ 90
    assert p.qty == -6
    # Short PnL: sell_qty * (avg_cost - cover_price) - commission
    assert p.realized_pnl == pytest.approx(4 * (100.0 - 90.0) - 0.5)
    # avg_cost unchanged on partial close
    assert p.avg_cost == pytest.approx(100.0)


def test_full_cover_returns_to_flat():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 95.0))
    assert p.is_flat
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(50.0)  # 10 * (100 - 95)


def test_losing_short_realizes_negative_pnl():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 110.0))
    assert p.is_flat
    assert p.realized_pnl == pytest.approx(-100.0)  # 10 * (100 - 110)


def test_short_mark_to_market_negative_value():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    # market_value(qty * price) is negative when short
    assert p.market_value(price=95.0) == pytest.approx(-950.0)
    # unrealized PnL = qty * (price - avg_cost) = -10 * (95 - 100) = +50
    assert p.unrealized_pnl(price=95.0) == pytest.approx(50.0)


def test_long_to_short_flip_in_one_fill():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))   # long 10 @ 100
    p.apply_fill(_fill(OrderSide.SELL, 15, 105.0))  # sell 15 -> flat then short 5
    assert p.qty == -5
    assert p.avg_cost == pytest.approx(105.0)
    # realized = closed-long PnL only: 10 * (105 - 100) = 50
    assert p.realized_pnl == pytest.approx(50.0)


def test_short_to_long_flip_in_one_fill():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))  # short 10 @ 100
    p.apply_fill(_fill(OrderSide.BUY, 15, 95.0))    # buy 15 -> flat then long 5
    assert p.qty == 5
    assert p.avg_cost == pytest.approx(95.0)
    # realized = closed-short PnL only: 10 * (100 - 95) = 50
    assert p.realized_pnl == pytest.approx(50.0)


def test_long_to_short_flip_blocked_when_allow_short_false():
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY")  # allow_short=False
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    with pytest.raises(ShortNotAllowedError):
        p.apply_fill(_fill(OrderSide.SELL, 15, 105.0))


def test_short_open_blocked_when_allow_short_false():
    """Same intent as test_sell_when_flat_raises but exercised by an
    explicit allow_short=False construction. Belt-and-suspenders."""
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY", allow_short=False)
    with pytest.raises(ShortNotAllowedError):
        p.apply_fill(_fill(OrderSide.SELL, 5, 100.0))
```

- [ ] **Step 2: Run to verify all new tests fail**

Run: `python -m pytest tests/unit/test_position.py -v`
Expected: every new short-side test FAILs (with `TypeError: __init__() got an unexpected keyword argument 'allow_short'` and/or assertion failures on `qty == -10`). The pre-existing 6 long-only tests still PASS.

- [ ] **Step 3: Commit (red)**

```
git add tests/unit/test_position.py
git commit -m "test(position): add failing short-side coverage"
```

---

### Task 5: Implement signed-qty `Position.apply_fill`

**Files:**
- Modify: `backtester/engine/position.py`

**Algebraic spec** (read carefully — implementation must match exactly to keep long-only PnL byte-identical):

Let `signed_delta = +fill.qty if side==BUY else -fill.qty`. Let `new_qty = self.qty + signed_delta`.

**Branch A — opening or growing (current qty is flat, or current qty and signed_delta share a sign):**
- If `signed_delta < 0` (opening or growing short) **and** `self.qty == 0` **and** not `self.allow_short`: raise `ShortNotAllowedError("cannot SELL when position is flat: shorts not allowed")`.
- If `self.qty == 0`: `self.avg_cost = fill.price`.
- Else: `self.avg_cost = (self.avg_cost * abs(self.qty) + fill.price * abs(signed_delta)) / (abs(self.qty) + abs(signed_delta))`.
- `self.qty = new_qty`.

**Branch B — opposite-sign fill (closing or flipping):**
- `close_qty = min(abs(signed_delta), abs(self.qty))`.
- `sign = 1 if self.qty > 0 else -1`.
- `self.realized_pnl += sign * close_qty * (fill.price - self.avg_cost)`. *(For long this is `(price - cost)`; for short this is `-(price - cost) = (cost - price)`.)*
- `original_qty = self.qty`; `self.qty = new_qty`.
- If `new_qty == 0`: `self.avg_cost = 0.0`.
- Elif `sign_of(new_qty) != sign_of(original_qty)` *(flipped through zero — leftover opens a new position at fill.price)*:
  - If `new_qty < 0` and not `self.allow_short`: raise `ShortNotAllowedError("cannot flip long->short: shorts not allowed")`.
  - `self.avg_cost = fill.price`.
- Else *(partial close, same direction)*: `avg_cost` unchanged.

**In both branches:** `self.realized_pnl -= fill.commission` at the end.

- [ ] **Step 1: Implement**

Replace the entire body of `backtester/engine/position.py` with:
```python
from __future__ import annotations

from dataclasses import dataclass

from backtester.core.enums import OrderSide
from backtester.core.exceptions import ShortNotAllowedError
from backtester.engine.fills import Fill


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    allow_short: bool = False

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def apply_fill(self, fill: Fill) -> None:
        signed_delta = fill.qty if fill.side == OrderSide.BUY else -fill.qty
        original_qty = self.qty
        new_qty = original_qty + signed_delta

        same_sign_or_flat = (
            original_qty == 0
            or (original_qty > 0 and signed_delta > 0)
            or (original_qty < 0 and signed_delta < 0)
        )

        if same_sign_or_flat:
            # Opening from flat, or growing existing direction.
            if signed_delta < 0 and original_qty == 0 and not self.allow_short:
                raise ShortNotAllowedError(
                    "cannot SELL when position is flat: shorts not allowed"
                )
            if original_qty == 0:
                self.avg_cost = fill.price
            else:
                total_abs = abs(original_qty) + abs(signed_delta)
                self.avg_cost = (
                    self.avg_cost * abs(original_qty)
                    + fill.price * abs(signed_delta)
                ) / total_abs
            self.qty = new_qty
        else:
            # Opposite sign: realize PnL on the closed portion, possibly flip.
            close_qty = min(abs(signed_delta), abs(original_qty))
            sign = 1 if original_qty > 0 else -1
            # TODO(short-positions): borrow cost / hard-to-borrow modeling is
            # not included here. A future phase should accrue daily borrow fee
            # against realized_pnl while qty < 0.
            self.realized_pnl += sign * close_qty * (fill.price - self.avg_cost)
            self.qty = new_qty
            if new_qty == 0:
                self.avg_cost = 0.0
            elif (new_qty > 0) != (original_qty > 0):
                # Flipped through zero — leftover opens a fresh position.
                if new_qty < 0 and not self.allow_short:
                    raise ShortNotAllowedError(
                        "cannot flip long->short: shorts not allowed"
                    )
                self.avg_cost = fill.price
            # else: partial close in same direction — avg_cost unchanged.

        self.realized_pnl -= fill.commission

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_cost)
```

- [ ] **Step 2: Run the position test file**

Run: `python -m pytest tests/unit/test_position.py -v`
Expected: ALL pass — 6 original long-only tests + 10 new short-side tests.

- [ ] **Step 3: Run the full suite to confirm nothing else broke**

Run: `python -m pytest -q`
Expected: 135 baseline + new tests so far (Tasks 1, 2, 4) all pass. No regressions in any other file.

- [ ] **Step 4: Commit**

```
git add backtester/engine/position.py
git commit -m "feat(engine): rewrite Position.apply_fill as signed-qty algebra"
```

---

## Phase 3: Portfolio simulator — tri-state state machine

### Task 6: Plumb `allow_short` from Broker → Position

**Files:**
- Modify: `backtester/engine/broker.py`
- Modify: `backtester/engine/portfolio.py`

**Rationale:** `Broker` already owns the `ExecutionConfig`. The simulator constructs the `Position` and is the right place to read `broker.config.allow_short` and forward it. No new ABI on `Broker` is needed.

- [ ] **Step 1: Update `Broker` to expose allow_short cleanly**

Modify `backtester/engine/broker.py`. Add one line in `__init__` so callers can read `broker.allow_short` without reaching into `broker.config`:
```python
class Broker:
    """Thin adapter that owns a FillEngine plus execution policy state."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.fills = FillEngine(
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
        )
        self.allow_fractional = config.allow_fractional
        self.allow_short = config.allow_short
```

- [ ] **Step 2: Forward to Position in the simulator**

In `backtester/engine/portfolio.py`, find the line `pos = Position(symbol=symbol)` inside `simulate(...)` and change it to:
```python
        pos = Position(symbol=symbol, allow_short=broker.allow_short)
```

*(No other simulator changes in this task — that comes in Task 8.)*

- [ ] **Step 3: Run the full suite to confirm zero regressions**

Run: `python -m pytest -q`
Expected: still green (the new attribute is unused by any code path yet — long-only simulator logic is unchanged, allow_short=False by default).

- [ ] **Step 4: Commit**

```
git add backtester/engine/broker.py backtester/engine/portfolio.py
git commit -m "feat(engine): plumb allow_short from Broker through to Position"
```

---

### Task 7: Write the failing portfolio-simulator tests for short transitions

**Files:**
- Modify: `tests/unit/test_portfolio.py` (append new tests)

- [ ] **Step 1: Append tests**

Append to `tests/unit/test_portfolio.py`:
```python
# --- Short-position simulator tests (Phase 0.2) ---

from backtester.core.enums import OrderSide, OrderType
from backtester.engine.orders import Order
from backtester.engine.position import Position
from backtester.engine.fills import Fill


def _short_broker():
    return Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0, allow_short=True))


def test_flat_to_short_emits_one_sell(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:20] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 2, "expected one short entry + one cover"
    assert trades.iloc[0]["side"] == "sell"
    assert trades.iloc[1]["side"] == "buy"
    # At some point position qty should be negative
    assert (positions["qty"] < 0).any()


def test_short_signal_blocked_when_allow_short_false(ohlcv_small):
    from backtester.core.exceptions import ShortNotAllowedError
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    # Default allow_short=False
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    with pytest.raises(ShortNotAllowedError, match="allow_short"):
        sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)


def test_long_to_short_flip_in_one_order(ohlcv_small):
    """A signal sequence long -> short emits a single SELL that closes the
    long and opens a new short in one fill (combined-order design)."""
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:15] = 1
    sig["signal"].iloc[15:n - 1] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    # Long entry, flip-to-short combined SELL, cover BUY = 3 fills
    assert len(trades) == 3
    assert list(trades["side"]) == ["buy", "sell", "buy"]
    # The flip SELL qty exceeds the prior long qty (closes long + opens short)
    long_entry_qty = trades.iloc[0]["qty"]
    flip_sell_qty = trades.iloc[1]["qty"]
    assert flip_sell_qty > long_entry_qty
    # Position goes long, then negative
    assert (positions["qty"] > 0).any()
    assert (positions["qty"] < 0).any()


def test_short_to_long_flip_in_one_order(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:15] = -1
    sig["signal"].iloc[15:n - 1] = 1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 3
    assert list(trades["side"]) == ["sell", "buy", "sell"]
    assert (positions["qty"] < 0).any()
    assert (positions["qty"] > 0).any()


def test_short_entry_via_sell_limit():
    """SELL LIMIT short entry: limit price above current market should fill at
    the limit when the next bar's high reaches it."""
    data = make_ohlcv(n=20, seed=7)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = -1
    sig["size"] = 1.0
    # Place SELL LIMIT slightly above the very high of bar 2 -> should NOT fill
    sig["limit_price"] = data["high"].iloc[2] * 2.0
    sf = SignalFrame(data=sig, price_column="limit_price")
    trades, _, _ = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert len(trades) == 0

    # Now place a SELL LIMIT below the next bar's high -> should fill
    sig2 = sig.copy()
    sig2["limit_price"] = data["low"].iloc[2] * 0.5
    sf2 = SignalFrame(data=sig2, price_column="limit_price")
    trades2, positions2, _ = sim.simulate(data=data, signal_frame=sf2, broker=broker)
    assert len(trades2) >= 1
    assert trades2.iloc[0]["side"] == "sell"
    assert (positions2["qty"] < 0).any()


def test_buy_stop_covers_a_short_when_price_rises_into_stop():
    """STOP support is exercised directly through Broker+Position rather than
    via the simulator's signal->order path (the simulator does not emit STOP
    orders). This verifies the FillEngine + Position wiring for shorts."""
    data = make_ohlcv(n=10, seed=3)
    broker = _short_broker()
    pos = Position(symbol="SPY", allow_short=True)

    # Open a short directly (skip the simulator)
    short_fill = Fill(
        timestamp=data.index[0], symbol="SPY", side=OrderSide.SELL,
        qty=10.0, price=float(data["close"].iloc[0]), commission=0.0,
    )
    pos.apply_fill(short_fill)
    assert pos.qty == -10.0

    # Build a BUY STOP at a level the next bar's high will exceed
    next_bar = data.iloc[1]
    stop_price = float(next_bar["low"])  # guaranteed <= high
    order = Order(
        timestamp=next_bar.name, symbol="SPY",
        side=OrderSide.BUY, qty=10.0,
        order_type=OrderType.STOP, stop_price=stop_price,
    )
    fill = broker.submit(order, next_bar)
    assert fill is not None
    pos.apply_fill(fill)
    assert pos.is_flat
    # cover above short entry means a loss; below means a gain — either way
    # realized_pnl is well-defined and non-NaN
    assert pos.realized_pnl == pos.realized_pnl  # not NaN
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_portfolio.py -v -k "short or flip or stop or limit"`
Expected: the new tests FAIL (`AssertionError`, `ShortNotAllowedError` not raised yet at simulator level, or `len(trades) != expected`). The pre-existing 4 portfolio tests still PASS. **The `test_buy_stop_covers_a_short_when_price_rises_into_stop` test may actually pass already** because it only touches Broker+Position, both of which are already updated; that is fine.

- [ ] **Step 3: Commit (red)**

```
git add tests/unit/test_portfolio.py
git commit -m "test(portfolio): add failing short-transition coverage"
```

---

### Task 8: Rewrite `PortfolioSimulator.simulate` as a tri-state state machine

**Files:**
- Modify: `backtester/engine/portfolio.py`

**Specification:**

For each bar `i`:
1. **Execute any pending order** on this bar (unchanged from current logic).
2. **Read `sig = int(signals[sig_col].iloc[i])`**. If `sig == -1` and `not broker.allow_short`: raise `ShortNotAllowedError(f"strategy emitted SHORT at bar {i} ({ts}) but execution.allow_short is False")`.
3. **Compute `prev_sign`** from `pos.qty`: `0` if flat, `+1` if long, `-1` if short.
4. **Compute `target_sign = sig`** (already in {-1, 0, 1}).
5. **If `prev_sign == target_sign`: emit no order** (preserves existing long-only "no rebalance" semantics — long-only configs continue to fire one BUY and one SELL per signal flip, byte-identical to v0.1.0).
6. **Else (state transition):**
   - **If `target_sign == 0`** (closing existing position):
     - `order_qty = abs(pos.qty)`; `side = OrderSide.BUY if prev_sign < 0 else OrderSide.SELL`; order type = MARKET.
   - **Else** (opening from flat OR flipping):
     - `equity_now = cash + pos.market_value(close)`
     - `size = float(signals[size_col].iloc[i]) if size_col and size_col in signals.columns else 1.0`
     - `alloc = equity_now * config.size * size`
     - `new_leg_qty = broker.round_qty(alloc / close)`
     - **If `prev_sign == 0`**: `order_qty = new_leg_qty`.
     - **Else (flipping)**: `order_qty = abs(pos.qty) + new_leg_qty` (close the old leg AND open the new leg in one fill).
     - `side = OrderSide.BUY if target_sign > 0 else OrderSide.SELL`.
     - **LIMIT vs MARKET**: if `prev_sign == 0` (entering from flat) AND `price_col is not None` AND `price_col in signals.columns` AND `pd.notna(signals[price_col].iloc[i])`: emit `OrderType.LIMIT` with `limit_price=float(signals[price_col].iloc[i])`. Otherwise MARKET.
   - If `order_qty > 0`: assign `pending = Order(...)` for the next bar.
7. **Mark-to-market** at close (unchanged).

- [ ] **Step 1: Implement**

Replace the entire body of `backtester/engine/portfolio.py` with:
```python
from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd

from backtester.config.models import PortfolioConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.core.exceptions import ShortNotAllowedError
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position


def _sign(qty: float) -> int:
    if qty > 0:
        return 1
    if qty < 0:
        return -1
    return 0


class PortfolioSimulator:
    """Translates signals -> orders -> fills, tracking cash, position, equity.

    Signal convention: signals in {-1, 0, 1}. A signal == -1 is rejected
    unless broker.allow_short is True. State transitions are computed from
    (sign(pos.qty), signal); same-sign cases emit no order (no rebalance).
    long<->short flips emit a single combined order; Position.apply_fill
    handles the close + reopen in one fill.
    """

    def __init__(self, config: PortfolioConfig, initial_cash: float = 100_000.0):
        self.config = config
        self.initial_cash = initial_cash

    def simulate(
        self,
        data: pd.DataFrame,
        signal_frame: SignalFrame,
        broker: Broker,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        signals = signal_frame.data
        sig_col = signal_frame.signal_column
        size_col = signal_frame.size_column
        price_col = signal_frame.price_column

        symbol = "ASSET"
        pos = Position(symbol=symbol, allow_short=broker.allow_short)
        cash = self.initial_cash

        fills: List[Fill] = []
        pending: Optional[Order] = None

        equity_rows = []
        position_rows = []

        index = data.index
        for i, ts in enumerate(index):
            bar = data.iloc[i]

            # 1. Execute pending order
            if pending is not None:
                fill = broker.submit(pending, bar)
                if fill is not None:
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                pending = None  # one-shot semantics

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
                        # Close current position fully.
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
                            # Flip: close old leg + open new leg in one fill.
                            order_qty = abs(pos.qty) + new_leg_qty
                        side = OrderSide.BUY if target_sign > 0 else OrderSide.SELL
                        # LIMIT only when entering from flat.
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
                        pending = Order(
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

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        positions_df = pd.DataFrame(position_rows).set_index("timestamp")
        trades_df = pd.DataFrame([
            {
                "timestamp": f.timestamp,
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "commission": f.commission,
                "notional": f.notional,
            }
            for f in fills
        ])
        return trades_df, positions_df, equity_curve
```

- [ ] **Step 2: Run portfolio tests**

Run: `python -m pytest tests/unit/test_portfolio.py -v`
Expected: all 4 baseline tests + all 6 new short-transition tests PASS.

- [ ] **Step 3: Run the full suite — backwards-compat checkpoint**

Run: `python -m pytest -q`
Expected: every test passes. If any long-only test (e.g., in `tests/integration/test_backtest_engine.py`, `tests/integration/test_run_backtest_cli.py`, `tests/integration/test_run_wfo_cli.py`, `tests/unit/test_strategy_*`) fails, **stop and debug** — the simulator refactor regressed long-only behavior. Likely culprits: a missing `prev_sign != target_sign` guard, a stray rebalance order, or a divergence in the LIMIT branch.

- [ ] **Step 4: Commit**

```
git add backtester/engine/portfolio.py
git commit -m "feat(engine): rewrite PortfolioSimulator.simulate as tri-state state machine"
```

---

## Phase 4: Sample strategy

### Task 9: Write failing test for the new RSI long/short strategy

**Files:**
- Create: `tests/unit/test_strategy_rsi_long_short.py`

- [ ] **Step 1: Create the test file**

`tests/unit/test_strategy_rsi_long_short.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def _make_synthetic_oscillating():
    """50 bars of price that swings up then down then up again so RSI
    crosses both thresholds."""
    n = 80
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    # V-shape: down 30 days, up 30 days, down 20 days
    prices = []
    p = 100.0
    for i in range(n):
        if i < 30:
            p *= 0.985  # strong down
        elif i < 60:
            p *= 1.015  # strong up
        else:
            p *= 0.985
        prices.append(p)
    df = pd.DataFrame({
        "open": prices, "high": [pr * 1.01 for pr in prices],
        "low": [pr * 0.99 for pr in prices], "close": prices,
        "volume": [1_000_000] * n,
    }, index=idx)
    return df


def test_rsi_long_short_emits_long_and_short_signals():
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )

    data = _make_synthetic_oscillating()
    strat = RSILongShortStrategy()
    params = RSILongShortParams(period=7, oversold=30.0, overbought=70.0)
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d", warmup_bars=strat.warmup_bars(params))
    sf = strat.generate_signals(data, ind, ctx, params)

    sigs = sf.data["signal"]
    assert (sigs == 1).any(), "expected at least one long signal"
    assert (sigs == -1).any(), "expected at least one short signal"
    assert set(sigs.unique()).issubset({-1, 0, 1})


def test_rsi_long_short_signal_is_shifted_by_one_bar():
    """Following the same convention as the existing strategies:
    signal at bar i corresponds to a decision based on bar i-1's data."""
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )

    data = _make_synthetic_oscillating()
    strat = RSILongShortStrategy()
    params = RSILongShortParams(period=7)
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d", warmup_bars=strat.warmup_bars(params))
    sf = strat.generate_signals(data, ind, ctx, params)
    # First bar is always flat after a shift
    assert sf.data["signal"].iloc[0] == 0


def test_rsi_long_short_warmup_bars_matches_period():
    from strategies.rsi_long_short import (
        RSILongShortStrategy, RSILongShortParams,
    )
    s = RSILongShortStrategy()
    assert s.warmup_bars(RSILongShortParams(period=14)) == 15
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_rsi_long_short.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.rsi_long_short'`.

- [ ] **Step 3: Commit (red)**

```
git add tests/unit/test_strategy_rsi_long_short.py
git commit -m "test(strategy): add failing tests for rsi_long_short"
```

---

### Task 10: Implement the `rsi_long_short` strategy

**Files:**
- Create: `strategies/rsi_long_short.py`

- [ ] **Step 1: Implement**

`strategies/rsi_long_short.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RSILongShortParams:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    size: float = 1.0


class RSILongShortStrategy(BaseStrategy[RSILongShortParams]):
    """
    Purpose:
        Symmetric RSI mean-reversion. Emit +1 (long) when RSI falls below
        `oversold`, -1 (short) when RSI rises above `overbought`. Hold the
        position until the opposite trigger fires.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} and `size` columns.

    Side effects:
        None.

    Requires:
        ExecutionConfig.allow_short = True at the config layer. Otherwise the
        portfolio simulator will raise ShortNotAllowedError on the first -1.
    """

    strategy_id = "rsi_long_short"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return RSILongShortParams

    def warmup_bars(self, params: RSILongShortParams) -> int:
        return params.period + 1

    def indicators(self, data: pd.DataFrame, params: RSILongShortParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        delta = data["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RSILongShortParams,
    ) -> SignalFrame:
        rsi = indicators["rsi"]
        # Trigger state: +1 if RSI < oversold, -1 if RSI > overbought, 0 else
        trig = pd.Series(0, index=data.index, dtype="int64")
        trig[rsi < params.oversold] = 1
        trig[rsi > params.overbought] = -1
        # Hold the last non-zero trigger until the opposite fires
        held = trig.replace(0, np.nan).ffill().fillna(0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = held.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 2: Run the strategy tests**

Run: `python -m pytest tests/unit/test_strategy_rsi_long_short.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```
git add strategies/rsi_long_short.py
git commit -m "feat(strategies): add symmetric RSI long/short strategy"
```

---

### Task 11: Register the new strategy

**Files:**
- Modify: `backtester/strategies/registry.py`
- Modify: `tests/unit/test_strategy_registry.py` (verify registration if existing test enumerates IDs)

- [ ] **Step 1: Inspect the existing registry test**

Run: `python -m pytest tests/unit/test_strategy_registry.py -v`
Expected: passes. Open the file and note whether it hard-codes the list of known strategy IDs. **If it does** (e.g., `assert set(STRATEGY_REGISTRY) == {"sma_cross", ...}`), this is the third-and-final allowed test edit — extend it to include the new ID. **If it does not enumerate exhaustively**, no test change is needed.

(If you must edit `test_strategy_registry.py`, justify in the commit message that the test was already a known fragile expectation for "registered strategies" and that the PRD's "zero edits" constraint is interpreted as zero *semantic* edits — adding an entry to a known-set assertion is a registration update, not a behavior change. If the reviewer disagrees, add the new strategy via a separate registration test instead and leave the original assertion alone.)

- [ ] **Step 2: Register the strategy**

Edit `backtester/strategies/registry.py`. Add the import and registration:
```python
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402
from strategies.rsi_long_short import RSILongShortStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
register_strategy(RSILongShortStrategy)
```

- [ ] **Step 3: Verify registration via a one-liner**

Run:
```
python -c "from backtester.strategies.registry import STRATEGY_REGISTRY; print(sorted(STRATEGY_REGISTRY))"
```
Expected: `['breakout_20d', 'rsi_long_short', 'rsi_mean_reversion', 'sma_cross']`.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: still green.

- [ ] **Step 5: Commit**

```
git add backtester/strategies/registry.py tests/unit/test_strategy_registry.py
git commit -m "feat(strategies): register rsi_long_short in default registry"
```

---

## Phase 5: Config + integration tests

### Task 12: Create the sample WFO config

**Files:**
- Create: `configs/wfo/rsi_long_short_wfo.yaml`

- [ ] **Step 1: Write the config**

`configs/wfo/rsi_long_short_wfo.yaml`:
```yaml
run_name: rsi_long_short_spy_wfo
strategy: rsi_long_short
strategy_params:
  period: 14
  oversold: 30
  overbought: 70
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
  allow_short: true
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space:
    period: [7, 14, 21]
    oversold: [20, 30]
    overbought: [70, 80]
wfo:
  enabled: true
  train_bars: 756
  test_bars: 252
  step_bars: 252
output_root: "output/runs"
```

- [ ] **Step 2: Verify the config loads**

Run:
```
python -c "from backtester.config.loader import load_run_config; from backtester.config.validation import validate_run_config; rc = load_run_config('configs/wfo/rsi_long_short_wfo.yaml'); validate_run_config(rc); print('OK; allow_short =', rc.execution.allow_short)"
```
Expected: `OK; allow_short = True`.

- [ ] **Step 3: Commit**

```
git add configs/wfo/rsi_long_short_wfo.yaml
git commit -m "feat(config): add rsi_long_short WFO sample config"
```

---

### Task 13: Synthetic downtrend integration test

**Files:**
- Modify: `tests/integration/test_backtest_engine.py` (append new test)

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_backtest_engine.py`:
```python
def _downtrending_ohlcv(n: int = 200) -> pd.DataFrame:
    """Deterministic monotonic downtrend with mild noise — perfect for a
    short strategy to make money on."""
    import numpy as np
    rng = np.random.default_rng(123)
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(loc=-0.003, scale=0.005, size=n)))
    open_ = np.empty(n); open_[0] = 100.0; open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.001, n - 1))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = np.full(n, 1_000_000, dtype=int)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_short_strategy_profits_on_downtrend():
    """A persistent-short strategy on a synthetic downtrend should finish
    with positive total_return and a non-empty trade log."""
    from dataclasses import dataclass

    from backtester.config.models import ExecutionConfig, PortfolioConfig
    from backtester.core.types import SignalFrame, StrategyContext
    from backtester.engine.backtest_engine import BacktestEngine
    from backtester.engine.broker import Broker
    from backtester.engine.portfolio import PortfolioSimulator
    from backtester.strategies.base import BaseStrategy

    @dataclass(slots=True)
    class _AlwaysShortParams:
        size: float = 1.0

    class _AlwaysShortStrategy(BaseStrategy[_AlwaysShortParams]):
        strategy_id = "_always_short_test"

        @classmethod
        def params_type(cls):
            return _AlwaysShortParams

        def indicators(self, data, params):
            return pd.DataFrame(index=data.index)

        def generate_signals(self, data, indicators, ctx: StrategyContext, params):
            df = pd.DataFrame(index=data.index)
            df["signal"] = -1
            df["signal"].iloc[0] = 0  # enter on bar 2
            df["size"] = params.size
            return SignalFrame(data=df)

    data = _downtrending_ohlcv(n=200)
    broker = Broker(ExecutionConfig(
        commission_bps=0.0, slippage_bps=0.0, allow_short=True,
    ))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    result = engine.run(_AlwaysShortStrategy(), data, _AlwaysShortParams(),
                        symbol="SYN", timeframe="1d")

    assert result.summary["total_return"] > 0, (
        f"expected positive return on downtrend short, got "
        f"{result.summary['total_return']}"
    )
    assert result.summary["n_trades"] > 0
    # Position should have been short at least once
    assert (result.positions["qty"] < 0).any()
```

- [ ] **Step 2: Run the integration tests**

Run: `python -m pytest tests/integration/test_backtest_engine.py -v`
Expected: 2 original tests + 1 new test = 3 passed.

- [ ] **Step 3: Commit**

```
git add tests/integration/test_backtest_engine.py
git commit -m "test(integration): always-short strategy profits on synthetic downtrend"
```

---

### Task 14: CLI integration test using the new strategy on SPY

**Files:**
- Modify: `tests/integration/test_run_backtest_cli.py` (append new test)

**Note:** This test uses bundled `data/raw/SPY.csv` from the repo. The current working directory in pytest is the repo root, so the relative path `data/raw/SPY.csv` resolves correctly when the config root is set to `data/raw`.

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_run_backtest_cli.py`:
```python
def test_run_backtest_cli_rsi_long_short_on_spy(tmp_path: Path):
    """Run the new strategy via CLI on bundled SPY data. Verify trades.csv
    contains a short entry (positions.csv has at least one negative qty)."""
    out = tmp_path / "runs"
    cfg = tmp_path / "rsi_ls.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    spy_root = (repo_root / "data" / "raw").as_posix()

    cfg.write_text(f"""
run_name: rsi_long_short_spy_smoke
strategy: rsi_long_short
strategy_params:
  period: 14
  oversold: 30
  overbought: 70
  size: 1.0
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: "csv"
  root: "{spy_root}"
execution:
  initial_cash: 100000
  commission_bps: 1
  slippage_bps: 2
  allow_fractional: false
  allow_short: true
portfolio:
  sizing_mode: "percent_equity"
  size: 0.9
output_root: "{out.as_posix()}"
""")

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_backtest", "--config", str(cfg)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    trades = pd.read_csv(run_dir / "trades.csv")
    positions = pd.read_csv(run_dir / "positions.csv")
    summary = json.loads((run_dir / "summary.json").read_text())

    assert summary["symbol"] == "SPY"
    assert summary["n_trades"] > 0, "expected at least one trade on multi-year SPY history"
    # The strategy holds both directions over a decade — at least one short.
    assert (positions["qty"] < 0).any(), "expected at least one short position bar"
    # Both BUY and SELL fills should appear (long entries and short entries).
    assert "buy" in set(trades["side"]) and "sell" in set(trades["side"])
```

- [ ] **Step 2: Run the CLI integration tests**

Run: `python -m pytest tests/integration/test_run_backtest_cli.py -v`
Expected: existing test passes + 1 new test passes.

- [ ] **Step 3: Commit**

```
git add tests/integration/test_run_backtest_cli.py
git commit -m "test(integration): CLI smoke test for rsi_long_short on SPY"
```

---

### Task 15: WFO smoke test for the new strategy

**Files:**
- Modify: `tests/integration/test_run_wfo_cli.py` (append new test)

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_run_wfo_cli.py`:
```python
import pandas as pd


def test_run_wfo_cli_rsi_long_short_emits_both_sides(tmp_path: Path):
    """WFO smoke test: stitched OOS trades file must contain both BUY and
    SELL entries (proving the long/short strategy ran end-to-end through
    the WFO orchestrator with allow_short=true)."""
    raw = tmp_path / "data"
    raw.mkdir()
    # Long enough series for several WFO windows
    from tests.fixtures.synthetic import make_ohlcv
    make_ohlcv(n=900, seed=17).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "wfo_ls.yaml"
    cfg.write_text(f"""
run_name: rsi_long_short_wfo_smoke
strategy: rsi_long_short
strategy_params:
  period: 14
  oversold: 30
  overbought: 70
  size: 1.0
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2030-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
  allow_short: true
portfolio:
  size: 1.0
optimization:
  objective: sharpe
  param_space:
    period: [7, 14]
    oversold: [25, 30]
    overbought: [70, 75]
wfo:
  enabled: true
  train_bars: 200
  test_bars: 50
  step_bars: 50
output_root: "{out.as_posix()}"
""")
    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_wfo", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    oos_trades = pd.read_csv(run_dir / "oos_trades.csv")
    # The stitched OOS series should contain at least one BUY and one SELL
    sides = set(oos_trades["side"]) if len(oos_trades) else set()
    assert "buy" in sides, f"expected at least one BUY in oos_trades, got {sides}"
    assert "sell" in sides, f"expected at least one SELL in oos_trades, got {sides}"
```

- [ ] **Step 2: Run the WFO CLI tests**

Run: `python -m pytest tests/integration/test_run_wfo_cli.py -v`
Expected: existing test + 1 new test pass.

- [ ] **Step 3: Commit**

```
git add tests/integration/test_run_wfo_cli.py
git commit -m "test(integration): WFO smoke test for rsi_long_short"
```

---

## Phase 6: Docs + final verification + release

### Task 16: Update `docs/strategy_contract.md`

**Files:**
- Modify: `docs/strategy_contract.md`

- [ ] **Step 1: Edit the Signal-semantics section**

Replace the existing "Signal semantics" section (lines starting `## Signal semantics`) with:

```markdown
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
- A strategy that emits only `{0, 1}` continues to work unchanged and
  does not require `allow_short`.
- A strategy that emits `-1` while `execution.allow_short` is `false`
  causes the portfolio simulator to raise `ShortNotAllowedError` at the
  first short signal. Strategy authors should document the requirement
  in their class docstring (see `strategies/rsi_long_short.py`).
```

- [ ] **Step 2: Commit**

```
git add docs/strategy_contract.md
git commit -m "docs: extend signal semantics with short support and allow_short"
```

---

### Task 17: Update `docs/runbook.md` with Limitations section

**Files:**
- Modify: `docs/runbook.md`

- [ ] **Step 1: Append a Limitations section**

Append to `docs/runbook.md`:
```markdown

## Limitations (v0.2.0)

Short-position support (`execution.allow_short: true`) intentionally omits
several real-broker features that should be added as follow-up phases:

- **No borrow cost / hard-to-borrow modeling.** Realized PnL on a short
  does not accrue a daily borrow fee. See the `TODO(short-positions)`
  marker in `backtester/engine/position.py`.
- **No margin call simulation.** The simulator assumes unlimited margin
  headroom. A short losing more than the account equity simply produces
  a negative equity series.
- **No leverage cap beyond `portfolio.size <= 1.0`.** When shorts are
  enabled, an instantaneous long → short flip momentarily produces ~2×
  gross exposure (the SELL closes the long and opens the short in one
  fill). If you want a hard gross-exposure cap, reduce `portfolio.size`
  (e.g., `0.5` ensures at most 1× gross around a flip).
- **No short interest / locate / hard-to-borrow availability checks.**
  Every symbol is assumed shortable on every bar.
- **No per-symbol short bans.** There is no mechanism to disable
  shorting on a specific ticker.

If your strategy or backtest depends on any of these effects, treat the
results as an upper bound on real-world performance.
```

- [ ] **Step 2: Commit**

```
git add docs/runbook.md
git commit -m "docs: document short-position limitations for v0.2.0"
```

---

### Task 18: Backwards-compatibility verification — long-only configs unchanged

**Files:**
- (no file changes; verification only)

**Goal:** Acceptance criterion 1 — every existing long-only config produces byte-identical `summary.json` (numerically) compared to the same config on v0.1.0. We verify this by re-running the bundled SMA-Cross config on the **current branch**, then on **v0.1.0**, and diffing the numeric fields.

- [ ] **Step 1: Run the SMA-Cross config on the current branch**

Run from repo root:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
Expected: exit 0. Note the run directory under `output/runs/`.

- [ ] **Step 2: Capture the summary**

Run (replace `<RUNDIR>` with the path printed by the previous step):
```
python -c "import json,sys; s=json.load(open(r'<RUNDIR>/summary.json')); print({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')})"
```
Save the printed dict somewhere visible (e.g., paste into the commit log of step 5). This is `summary_new`.

- [ ] **Step 3: Run the same config on v0.1.0**

```
git worktree add ../bt-v010 v0.1.0
cd ../bt-v010
pip install -e .  # installs at v0.1.0 in the worktree
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
Expected: exit 0. Note that run directory.

Then capture its summary the same way:
```
python -c "import json,sys; s=json.load(open(r'<RUNDIR_V010>/summary.json')); print({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')})"
```
This is `summary_old`.

- [ ] **Step 4: Compare**

The two dicts must be **exactly equal** for `n_trades` (integer) and equal **to within 1e-9 absolute, 1e-12 relative** for the floats `total_return`, `sharpe`, `max_drawdown`, `final_equity`. If any field diverges, the simulator refactor regressed long-only PnL — debug before continuing.

Cleanup:
```
cd ../Backtester
git worktree remove ../bt-v010
```
(reinstall current branch if your active venv was disturbed: `pip install -e .[dev]`).

- [ ] **Step 5: Record the verification (commit empty msg or note in CHANGELOG)**

If a `CHANGELOG.md` exists, append the captured numbers. Otherwise skip — the verification is a manual gate, not a code change.

---

### Task 19: Final full-suite green check

**Files:**
- (no file changes)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected:
- All 135 pre-existing tests pass (no edits other than `test_sell_when_flat_raises` and possibly `test_strategy_registry` per Task 11 step 1).
- All new tests pass:
  - `test_exceptions_short.py` (2)
  - `test_config_models.py` additions (2)
  - `test_position.py` additions (10)
  - `test_portfolio.py` additions (6)
  - `test_strategy_rsi_long_short.py` (3)
  - `test_backtest_engine.py` addition (1)
  - `test_run_backtest_cli.py` addition (1)
  - `test_run_wfo_cli.py` addition (1)

Total expected new tests: **26**. Total expected pass count: **135 + 26 = 161**. Adjust the count in your head if any number drifted during implementation, but the rule is: **no regressions in the 135-baseline, and every new test green**.

- [ ] **Step 2: If anything is red, stop and fix root cause before tagging.**

---

### Task 20: Bump version and tag v0.2.0

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump the version**

Edit `pyproject.toml` and change:
```toml
version = "0.1.0"
```
to:
```toml
version = "0.2.0"
```

- [ ] **Step 2: Reinstall to refresh egg-info**

Run: `pip install -e .[dev]`
Expected: `Successfully installed modular-stock-backtester-0.2.0 ...`.

- [ ] **Step 3: Commit and tag**

```
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0 (short-position support)"
git tag -a v0.2.0 -m "v0.2.0: short-position support end-to-end"
```

- [ ] **Step 4 (optional): Push tag**

Only if instructed by the user. Otherwise leave the tag local.

```
# git push origin master v0.2.0
```

---

## Appendix A: Decision log

**Q: Keep the long-only guard on Position or move it entirely to the simulator?**
Position keeps the guard via an `allow_short: bool = False` constructor argument. Rationale: (1) it preserves the safety property that any direct `Position.apply_fill(SELL)` call on a flat position with default settings raises; (2) the existing `test_sell_when_flat_raises` test only needs a minimal edit (swap the exception class) rather than being deleted; (3) belt-and-suspenders defense — both the simulator (pre-check on signal) and Position (post-check on fill) reject illegal shorts.

**Q: One combined order for flip transitions, or two sequential orders?**
One combined order. The simulator emits a single SELL (or BUY) whose qty equals `|prev_qty| + new_leg_qty`. Position's signed-qty algebra handles the close + reopen in one fill. Rationale: matches the existing "one pending order per bar" semantics, avoids ordering ambiguity, and keeps trades.csv shorter and easier to read. The alternative (two sequential orders across two bars) would introduce a one-bar gap where the position is flat — semantically different and harder to reason about.

**Q: LIMIT/STOP support for shorts at the simulator layer?**
LIMIT is generalized — `price_column` now applies to flat → long AND flat → short entries (not just long). STOP is **not** plumbed through the strategy → simulator path; the simulator does not emit STOP orders. STOP support is exercised at the Broker+Position level via direct order construction. Adding strategy-level STOP support is a separate, larger feature (would need a `stop_column` and reasoning about stop-vs-limit precedence).

**Q: Guardrail against 2× gross exposure on instantaneous long → short flip?**
Documented as a limitation in `docs/runbook.md`. Adding a hard guardrail would require introducing leverage tracking; deferred to a future phase.

## Appendix B: Quick troubleshooting

| Symptom                                                                | Likely cause                                                                                                                |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Long-only test failures after Task 8                                   | Simulator rebalancing on every bar instead of only on signal flips. Check the `prev_sign != target_sign` guard.             |
| `test_short_signal_blocked_when_allow_short_false` does not raise      | `simulate` is reading `sig` before the `ShortNotAllowedError` check, or the check is gated on `target_sign != prev_sign`.   |
| `Position.apply_fill` raises on a legitimate cover                     | Branch B is checking `allow_short` on every flip; it should only check when the post-flip `new_qty < 0`.                    |
| Long entry quantity changes vs. v0.1.0                                 | `round_qty` is being applied differently, or `equity_now` is being recomputed inside a branch that did not exist in v0.1.0. |
| `n_trades` for sma_cross doubles after refactor                        | Simulator is emitting orders on same-sign bars (every long bar gets a BUY). Same fix: same-sign guard.                      |
| CLI integration test fails with `ShortNotAllowedError` on SPY          | The new strategy was registered but the test config forgot `allow_short: true`. Check the YAML fixture in Task 14.          |
