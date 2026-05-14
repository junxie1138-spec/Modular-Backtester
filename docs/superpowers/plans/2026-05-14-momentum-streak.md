# `momentum_streak` Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a basic symmetric long/short momentum strategy (`momentum_streak`) that takes positions after a configurable run of consecutive same-direction days confirmed by above-average volume. Wire it end-to-end through single backtest, grid optimization, and walk-forward optimization (WFO).

**Architecture:** One new strategy module (`strategies/momentum_streak.py`) following the same `BaseStrategy[Params]` pattern as the existing four strategies. Vectorized indicators (streak counts via groupby-cumcount, rolling-mean volume confirmation) plus one O(n) state-machine loop for the tri-state signal. Three new YAML configs (one per workflow). One new unit test file plus two appended integration smoke tests. Zero framework, engine, broker, or simulator changes — the v0.2.0 tri-state simulator already handles every transition this strategy emits, including direct long→short flips.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new runtime dependencies.

**Scope notes:**
- Symmetric long/short. Requires `execution.allow_short: true` in any config that wants the strategy to take shorts.
- Exits do NOT require volume confirmation (per design spec §1.2).
- Doji days (`close == prev_close`) reset both streak counters to 0 (per design spec §1.1).
- No LIMIT-order entries (MARKET only). No price-column on the SignalFrame.
- Parameter validation is not enforced at the dataclass layer (matches the project convention — degenerate values produce empty signals, not exceptions).

**Required reading before starting:**
1. `docs/superpowers/specs/2026-05-14-momentum-streak-design.md` — the approved design spec this plan implements.
2. `docs/strategy_contract.md` — strategy interface, signal semantics including the `-1` value added in v0.2.0.
3. `strategies/rsi_long_short.py` — closest reference: a v0.2.0 long/short strategy that emits `{-1, 0, 1}` with `shift(1)`.
4. `strategies/breakout_20d.py` — reference for the state-machine-via-forward-fill pattern (we use an explicit loop instead, but the file conventions match).
5. `backtester/strategies/registry.py` — explicit registration list.
6. `tests/unit/test_strategy_rsi_long_short.py` — reference test style for symmetric long/short strategies.
7. `tests/integration/test_run_backtest_cli.py` and `tests/integration/test_run_wfo_cli.py` — CLI smoke-test patterns.

**Baseline verification before starting:**

Run from repo root:
```
python -m pytest -q
```
Expected: `162 passed` (the v0.2.0 baseline after the WFO stitcher fix). Record the exact count — every existing test must still pass at the end.

**File-structure preview:**

| File | Action | Purpose |
|---|---|---|
| `strategies/momentum_streak.py` | create | Strategy module + params dataclass |
| `backtester/strategies/registry.py` | modify | Register `momentum_streak` |
| `configs/backtests/momentum_streak_spy.yaml` | create | Single-backtest sample config |
| `configs/optimize/momentum_streak_grid.yaml` | create | Grid-search sample config (81 combos) |
| `configs/wfo/momentum_streak_wfo.yaml` | create | WFO sample config |
| `tests/unit/test_strategy_momentum_streak.py` | create | 8 unit tests |
| `tests/integration/test_run_backtest_cli.py` | append 1 test | CLI smoke on SPY |
| `tests/integration/test_run_wfo_cli.py` | append 1 test | WFO CLI smoke on synthetic data |

---

## Phase 1: Strategy module (TDD)

### Task 1: Write the failing unit tests

**Files:**
- Create: `tests/unit/test_strategy_momentum_streak.py`

**Rationale:** Eight tests that lock in the contract for the new strategy. They will all fail with `ModuleNotFoundError` until Task 2 lands.

- [ ] **Step 1: Create the test file**

`tests/unit/test_strategy_momentum_streak.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def _ohlcv_from_closes(closes, volumes=None):
    """Build a minimal OHLCV frame from a close-price sequence.
    Open/high/low are derived deterministically so the validators don't choke.
    If `volumes` is None, fills with a constant 1_000_000 per bar."""
    n = len(closes)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = np.asarray(closes, dtype=float)
    if volumes is None:
        volumes = np.full(n, 1_000_000, dtype=float)
    else:
        volumes = np.asarray(volumes, dtype=float)
    opens = closes.copy()
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=idx)


def _run(strat, params, data):
    """Helper: build indicators + ctx + signals in one call."""
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d",
                          warmup_bars=strat.warmup_bars(params))
    return strat.generate_signals(data, ind, ctx, params)


def test_green_streak_resets_on_red():
    """Closes +1, +1, +1, -1, +1 -> green_streak series 1, 2, 3, 0, 1."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    closes = [100.0, 101.0, 102.0, 103.0, 102.0, 103.0]
    # First bar's diff is NaN -> counts as neither green nor red.
    data = _ohlcv_from_closes(closes)
    ind = MomentumStreakStrategy().indicators(data, MomentumStreakParams())
    # Expected green_streak: 0 (first bar, no prev), 1, 2, 3, 0, 1
    assert list(ind["green_streak"]) == [0, 1, 2, 3, 0, 1]
    # Expected red_streak: 0, 0, 0, 0, 1, 0
    assert list(ind["red_streak"]) == [0, 0, 0, 0, 1, 0]


def test_streak_resets_on_doji():
    """Closes +1, =, +1 -> green streak series 1, 0, 1; red streak 0, 0, 0."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    closes = [100.0, 101.0, 101.0, 102.0]
    data = _ohlcv_from_closes(closes)
    ind = MomentumStreakStrategy().indicators(data, MomentumStreakParams())
    assert list(ind["green_streak"]) == [0, 1, 0, 1]
    assert list(ind["red_streak"]) == [0, 0, 0, 0]


def test_long_entry_fires_on_streak_plus_volume():
    """3 greens + above-average volume on the 3rd green -> signal flips to +1
    on the bar AFTER (one-bar shift)."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    # Need >= vol_lookback bars of warmup. Use vol_lookback=3, entry_streak=3.
    # Bars 0..2: flat (vol_sma warming up). Bars 3..5: three greens with high vol.
    closes = [100.0, 100.0, 100.0,  # warmup, flat
              101.0, 102.0, 103.0,  # 3 greens
              103.5]                # extra bar to observe the shifted signal
    volumes = [1_000_000] * 3 + [5_000_000] * 3 + [1_000_000]
    data = _ohlcv_from_closes(closes, volumes)
    params = MomentumStreakParams(entry_streak=3, exit_streak=2,
                                  vol_lookback=3, vol_mult=1.0)
    sf = _run(MomentumStreakStrategy(), params, data)
    sigs = list(sf.data["signal"])
    # The third green is at index 5; entry fires there; shifted signal is +1 at index 6.
    assert sigs[6] == 1
    # Earlier bars are 0 (warmup / streak not yet long enough / shifted from None).
    assert all(s == 0 for s in sigs[:6])


def test_long_entry_suppressed_when_volume_below_threshold():
    """Same streak as above but volume on the 3rd green is BELOW the SMA -> no entry."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    closes = [100.0, 100.0, 100.0,
              101.0, 102.0, 103.0,
              103.5]
    # Volume on bar 5 (the 3rd green) is *below* the trailing-3 SMA.
    volumes = [5_000_000, 5_000_000, 5_000_000,
               1_000_000, 1_000_000, 100_000,
               1_000_000]
    data = _ohlcv_from_closes(closes, volumes)
    params = MomentumStreakParams(entry_streak=3, exit_streak=2,
                                  vol_lookback=3, vol_mult=1.0)
    sf = _run(MomentumStreakStrategy(), params, data)
    assert all(s == 0 for s in sf.data["signal"])


def test_long_exit_after_exit_streak_reds():
    """Open long via volume-confirmed greens, then 2 reds -> signal returns to 0."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    closes = [100.0, 100.0, 100.0,            # warmup
              101.0, 102.0, 103.0,            # 3 greens (entry triggers at idx 5)
              104.0,                          # held long, still green (state stays +1)
              103.0, 102.0,                   # 2 reds -> exit triggers at idx 8
              102.0]                          # extra bar so we observe the shifted 0
    volumes = [1_000_000] * 3 + [5_000_000] * 3 + [1_000_000] * 4
    data = _ohlcv_from_closes(closes, volumes)
    params = MomentumStreakParams(entry_streak=3, exit_streak=2,
                                  vol_lookback=3, vol_mult=1.0)
    sf = _run(MomentumStreakStrategy(), params, data)
    sigs = list(sf.data["signal"])
    # Long entry shifted: signal becomes +1 at idx 6 onwards while held.
    assert sigs[6] == 1
    assert sigs[7] == 1  # still long during the in-progress reds (1 red, exit_streak=2)
    assert sigs[8] == 1  # second red is at idx 8; exit fires there; shift -> idx 9
    assert sigs[9] == 0


def test_short_entry_symmetric_to_long():
    """3 reds + above-average volume on the 3rd red -> signal becomes -1 after shift."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    closes = [100.0, 100.0, 100.0,    # warmup
              99.0, 98.0, 97.0,       # 3 reds
              97.0]                   # observation bar
    volumes = [1_000_000] * 3 + [5_000_000] * 3 + [1_000_000]
    data = _ohlcv_from_closes(closes, volumes)
    params = MomentumStreakParams(entry_streak=3, exit_streak=2,
                                  vol_lookback=3, vol_mult=1.0)
    sf = _run(MomentumStreakStrategy(), params, data)
    sigs = list(sf.data["signal"])
    assert sigs[6] == -1
    assert all(s == 0 for s in sigs[:6])


def test_direct_long_to_short_flip_on_opposite_high_volume_streak():
    """While holding long, three consecutive reds with high volume -> direct flip to -1
    (no intervening 0)."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    # Build: warmup, greens to go long, then reds-with-volume to flip to short.
    closes = [100.0, 100.0, 100.0,        # warmup (bars 0..2)
              101.0, 102.0, 103.0,        # 3 greens with vol -> long entry at bar 5
              104.0,                      # held long (bar 6)
              103.0, 102.0, 101.0,        # 3 reds with vol -> flip at bar 9
              101.0]                      # observation (bar 10)
    volumes = [1_000_000] * 3 + [5_000_000] * 3 + [1_000_000] + [5_000_000] * 3 + [1_000_000]
    data = _ohlcv_from_closes(closes, volumes)
    params = MomentumStreakParams(entry_streak=3, exit_streak=2,
                                  vol_lookback=3, vol_mult=1.0)
    sf = _run(MomentumStreakStrategy(), params, data)
    sigs = list(sf.data["signal"])
    # After shift: long active from idx 6 until the flip; flip occurs at bar 9
    # (third red), so shifted signal at idx 10 should be -1.
    # exit_streak=2 means the long would exit after 2 reds (at bar 8), but
    # entry_streak=3 also triggers a short on bar 9. Per the state-machine
    # in §1.2 the flip wins over the plain exit:
    #   bar 7: 1 red so far, prev state +1, no flip yet -> stays +1
    #   bar 8: 2 reds, prev state +1, short_entry not yet (need 3 reds with vol), long_exit fires -> 0
    #   bar 9: 3 reds, prev state 0, short_entry fires (vol high) -> -1
    # After shift:
    #   shifted signal at idx 8 corresponds to state at idx 7 = +1
    #   shifted signal at idx 9 corresponds to state at idx 8 = 0
    #   shifted signal at idx 10 corresponds to state at idx 9 = -1
    assert sigs[8] == 1
    assert sigs[9] == 0
    assert sigs[10] == -1


def test_warmup_and_first_bar_is_zero():
    """First bar signal is 0 (shift-by-one). warmup_bars == max(entry_streak,
    exit_streak, vol_lookback) + 1."""
    from strategies.momentum_streak import (
        MomentumStreakStrategy, MomentumStreakParams,
    )
    strat = MomentumStreakStrategy()
    assert strat.warmup_bars(MomentumStreakParams(entry_streak=3, exit_streak=2, vol_lookback=20)) == 21
    assert strat.warmup_bars(MomentumStreakParams(entry_streak=5, exit_streak=10, vol_lookback=3)) == 11
    data = _ohlcv_from_closes([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    sf = _run(strat, MomentumStreakParams(entry_streak=2, exit_streak=1, vol_lookback=3), data)
    assert sf.data["signal"].iloc[0] == 0
```

- [ ] **Step 2: Run the tests; verify they all fail**

Run: `python -m pytest tests/unit/test_strategy_momentum_streak.py -v`
Expected: 8 tests collected, all FAIL with `ModuleNotFoundError: No module named 'strategies.momentum_streak'`.

- [ ] **Step 3: Commit (red)**

```bash
git add tests/unit/test_strategy_momentum_streak.py
git commit -m "test(strategy): add failing tests for momentum_streak"
```

---

### Task 2: Implement `momentum_streak`

**Files:**
- Create: `strategies/momentum_streak.py`

**Algebraic spec for the indicators**:

For a series of closes `c[0..n-1]`:
- `up[i] = c[i] > c[i-1]` (False at i=0 because `c[i-1]` is NaN; `NaN > x → False` in pandas)
- `down[i] = c[i] < c[i-1]` (False at i=0)
- `green_streak[i] = number of consecutive True bars ending at i in 'up'; 0 if `up[i]` is False`
- `red_streak[i] = symmetric in 'down'`
- `vol_sma[i] = mean of volume[i-vol_lookback+1 .. i]` (NaN until i >= vol_lookback - 1)
- `vol_confirm[i] = volume[i] > vol_mult * vol_sma[i]` (False where `vol_sma` is NaN)

The streak counter must reset on every False bar, INCLUDING dojis (where both `up` and `down` are False). The vectorized recipe `(mask != mask.shift(1)).cumsum()` produces group ids per consecutive run; `groupby(group).cumcount() + 1` then gives 1, 2, 3, ... within the True runs; we zero it out on False bars with `.where(mask, 0)`.

**State machine (Task 2 step body — copy verbatim into the strategy)**:

```python
state = np.zeros(len(data), dtype=int)
for i in range(1, len(data)):
    prev = state[i - 1]
    le = bool(long_entry.iloc[i])
    se = bool(short_entry.iloc[i])
    lx = bool(long_exit.iloc[i])
    sx = bool(short_exit.iloc[i])
    if prev == 0:
        state[i] = 1 if le else (-1 if se else 0)
    elif prev == 1:
        if se:
            state[i] = -1
        elif lx:
            state[i] = 0
        else:
            state[i] = 1
    else:  # prev == -1
        if le:
            state[i] = 1
        elif sx:
            state[i] = 0
        else:
            state[i] = -1
```

- [ ] **Step 1: Implement the strategy module**

Create `strategies/momentum_streak.py` with the full content below (no placeholders, no missing imports):

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumStreakParams:
    entry_streak: int = 3
    exit_streak: int = 2
    vol_lookback: int = 20
    vol_mult: float = 1.0
    size: float = 1.0


def _consecutive_streak(mask: pd.Series) -> pd.Series:
    """Per-bar count of the current True run length in `mask`; 0 on False bars.

    Example:
        mask  = [F, T, T, T, F, T]
        out   = [0, 1, 2, 3, 0, 1]
    """
    mask = mask.fillna(False).astype(bool)
    grp = (mask != mask.shift(1)).cumsum()
    counts = mask.groupby(grp).cumcount().add(1)
    return counts.where(mask, 0).astype(int)


class MomentumStreakStrategy(BaseStrategy[MomentumStreakParams]):
    """
    Purpose:
        Symmetric momentum: enter LONG after `entry_streak` consecutive up-days
        (close > prev_close) confirmed by above-average volume on the entry
        bar; enter SHORT after `entry_streak` consecutive down-days similarly
        confirmed. Exit a long after `exit_streak` consecutive down-days;
        exit a short after `exit_streak` consecutive up-days. Doji days
        (close == prev_close) reset both streak counters to 0.

    Inputs:
        OHLCV dataframe with datetime index and `close`, `volume` columns.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} and `size` columns.

    Side effects:
        None.

    Requires:
        `execution.allow_short: true` in the run config for the short side to
        fire. Without it, the portfolio simulator raises ShortNotAllowedError
        on the first short signal.
    """

    strategy_id = "momentum_streak"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return MomentumStreakParams

    def warmup_bars(self, params: MomentumStreakParams) -> int:
        return max(params.entry_streak, params.exit_streak, params.vol_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: MomentumStreakParams) -> pd.DataFrame:
        close = data["close"]
        prev_close = close.shift(1)
        up = (close > prev_close).fillna(False)
        down = (close < prev_close).fillna(False)

        green_streak = _consecutive_streak(up)
        red_streak = _consecutive_streak(down)

        vol_sma = data["volume"].rolling(params.vol_lookback).mean()
        vol_confirm = (data["volume"] > params.vol_mult * vol_sma).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["up"] = up.astype(bool)
        out["down"] = down.astype(bool)
        out["green_streak"] = green_streak
        out["red_streak"] = red_streak
        out["vol_sma"] = vol_sma
        out["vol_confirm"] = vol_confirm.astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumStreakParams,
    ) -> SignalFrame:
        green_streak = indicators["green_streak"]
        red_streak = indicators["red_streak"]
        vol_confirm = indicators["vol_confirm"]

        long_entry = (green_streak >= params.entry_streak) & vol_confirm
        short_entry = (red_streak >= params.entry_streak) & vol_confirm
        long_exit = red_streak >= params.exit_streak
        short_exit = green_streak >= params.exit_streak

        state = np.zeros(len(data), dtype=int)
        for i in range(1, len(data)):
            prev = state[i - 1]
            le = bool(long_entry.iloc[i])
            se = bool(short_entry.iloc[i])
            lx = bool(long_exit.iloc[i])
            sx = bool(short_exit.iloc[i])
            if prev == 0:
                state[i] = 1 if le else (-1 if se else 0)
            elif prev == 1:
                if se:
                    state[i] = -1
                elif lx:
                    state[i] = 0
                else:
                    state[i] = 1
            else:  # prev == -1
                if le:
                    state[i] = 1
                elif sx:
                    state[i] = 0
                else:
                    state[i] = -1

        signal = pd.Series(state, index=data.index).shift(1).fillna(0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 2: Run the unit tests**

Run: `python -m pytest tests/unit/test_strategy_momentum_streak.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 3: If any test fails**, read the failure carefully. Likely culprits:
- Streak counter off-by-one: check that `_consecutive_streak` returns `1` on the first True bar of a run, not `0`.
- Volume confirm True during warmup: check `.fillna(False)` after `> vol_mult * vol_sma`.
- State at idx 0 not zero: the loop starts at `i=1` and leaves `state[0]=0` — verify.
- Direct flip test fails because long_exit fires before short_entry: re-check the state machine — from prev=+1, `se` is checked before `lx`. That's correct.

- [ ] **Step 4: Run the full suite — sanity check**

Run: `python -m pytest -q`
Expected: 162 baseline + 8 new = **170 passed**, zero regressions.

- [ ] **Step 5: Commit**

```bash
git add strategies/momentum_streak.py
git commit -m "feat(strategies): add symmetric momentum_streak strategy"
```

---

### Task 3: Register the strategy

**Files:**
- Modify: `backtester/strategies/registry.py`

- [ ] **Step 1: Add import and registration**

Edit `backtester/strategies/registry.py`. After the existing `RSILongShortStrategy` import and registration block, add the new strategy so the file ends with:

```python
# --- Default strategy registrations (explicit, predictable order) ---
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402
from strategies.rsi_long_short import RSILongShortStrategy  # noqa: E402
from strategies.momentum_streak import MomentumStreakStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
register_strategy(RSILongShortStrategy)
register_strategy(MomentumStreakStrategy)
```

- [ ] **Step 2: Verify registration via a one-liner**

Run:
```
python -c "from backtester.strategies.registry import STRATEGY_REGISTRY; print(sorted(STRATEGY_REGISTRY))"
```
Expected output (exactly):
```
['breakout_20d', 'momentum_streak', 'rsi_long_short', 'rsi_mean_reversion', 'sma_cross']
```

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: still 170 passed, zero regressions.

- [ ] **Step 4: Commit**

```bash
git add backtester/strategies/registry.py
git commit -m "feat(strategies): register momentum_streak in default registry"
```

---

## Phase 2: Sample configs

### Task 4: Single-backtest config

**Files:**
- Create: `configs/backtests/momentum_streak_spy.yaml`

- [ ] **Step 1: Write the config**

`configs/backtests/momentum_streak_spy.yaml`:
```yaml
run_name: momentum_streak_spy
strategy: momentum_streak
strategy_params:
  entry_streak: 3
  exit_streak: 2
  vol_lookback: 20
  vol_mult: 1.0
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
output_root: "output/runs"
```

- [ ] **Step 2: Verify the config loads and validates**

Run:
```
python -c "from backtester.config.loader import load_run_config; from backtester.config.validation import validate_run_config; rc = load_run_config('configs/backtests/momentum_streak_spy.yaml'); validate_run_config(rc); print('OK; strategy=', rc.strategy, 'allow_short=', rc.execution.allow_short)"
```
Expected: `OK; strategy= momentum_streak allow_short= True`.

- [ ] **Step 3: Smoke-run the CLI**

Run: `python -m backtester.runners.run_backtest --config configs/backtests/momentum_streak_spy.yaml`
Expected: exit 0, log lines including `done: total_return=...`. The actual numbers don't matter for this step — only that the run completes without error. (Output artifacts under `output/runs/` are gitignored, so they won't pollute the working tree.)

- [ ] **Step 4: Commit**

```bash
git add configs/backtests/momentum_streak_spy.yaml
git commit -m "feat(config): add momentum_streak SPY backtest config"
```

---

### Task 5: Grid-optimization config

**Files:**
- Create: `configs/optimize/momentum_streak_grid.yaml`

- [ ] **Step 1: Write the config**

`configs/optimize/momentum_streak_grid.yaml`:
```yaml
run_name: momentum_streak_spy_grid
strategy: momentum_streak
strategy_params:
  entry_streak: 3
  exit_streak: 2
  vol_lookback: 20
  vol_mult: 1.0
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
    entry_streak: [2, 3, 5]
    exit_streak:  [1, 2, 3]
    vol_lookback: [10, 20, 50]
    vol_mult:     [1.0, 1.25, 1.5]
output_root: "output/runs"
```

- [ ] **Step 2: Verify the config loads and validates**

Run:
```
python -c "from backtester.config.loader import load_run_config; from backtester.config.validation import validate_run_config; rc = load_run_config('configs/optimize/momentum_streak_grid.yaml'); validate_run_config(rc); print('OK; n_combos=', len(rc.optimization.param_space['entry_streak'])*len(rc.optimization.param_space['exit_streak'])*len(rc.optimization.param_space['vol_lookback'])*len(rc.optimization.param_space['vol_mult']))"
```
Expected: `OK; n_combos= 81`.

- [ ] **Step 3: Smoke-run the CLI**

Run: `python -m backtester.runners.run_optimize --config configs/optimize/momentum_streak_grid.yaml`
Expected: exit 0. The run will iterate 81 combinations on a 10-year SPY series; on a typical laptop this completes in under 30 seconds. Log line near the end should announce a best parameter set with a sharpe value.

- [ ] **Step 4: Commit**

```bash
git add configs/optimize/momentum_streak_grid.yaml
git commit -m "feat(config): add momentum_streak grid optimization config"
```

---

### Task 6: WFO config

**Files:**
- Create: `configs/wfo/momentum_streak_wfo.yaml`

- [ ] **Step 1: Write the config**

`configs/wfo/momentum_streak_wfo.yaml`:
```yaml
run_name: momentum_streak_spy_wfo
strategy: momentum_streak
strategy_params:
  entry_streak: 3
  exit_streak: 2
  vol_lookback: 20
  vol_mult: 1.0
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
    entry_streak: [2, 3, 5]
    exit_streak:  [1, 2, 3]
    vol_lookback: [10, 20, 50]
    vol_mult:     [1.0, 1.25, 1.5]
wfo:
  enabled: true
  train_bars: 756
  test_bars: 252
  step_bars: 252
output_root: "output/runs"
```

- [ ] **Step 2: Verify the config loads and validates**

Run:
```
python -c "from backtester.config.loader import load_run_config; from backtester.config.validation import validate_run_config; rc = load_run_config('configs/wfo/momentum_streak_wfo.yaml'); validate_run_config(rc); print('OK; wfo.enabled=', rc.wfo.enabled, 'train=', rc.wfo.train_bars)"
```
Expected: `OK; wfo.enabled= True train= 756`.

- [ ] **Step 3: Smoke-run the CLI**

Run: `python -m backtester.runners.run_wfo --config configs/wfo/momentum_streak_wfo.yaml`
Expected: exit 0. The run will iterate 6 WFO windows × 81 combinations each on a 10-year SPY series; typical runtime ~20–60 seconds. Log lines should announce each window's best params and final OOS sharpe.

- [ ] **Step 4: Commit**

```bash
git add configs/wfo/momentum_streak_wfo.yaml
git commit -m "feat(config): add momentum_streak WFO config"
```

---

## Phase 3: Integration tests

### Task 7: CLI backtest smoke test on SPY

**Files:**
- Modify: `tests/integration/test_run_backtest_cli.py` (append one test)

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_run_backtest_cli.py`:
```python
def test_run_backtest_cli_momentum_streak_on_spy(tmp_path: Path):
    """Run momentum_streak via the CLI on bundled SPY data. Verify both BUY
    and SELL fills appear and at least one bar shows a negative position."""
    out = tmp_path / "runs"
    cfg = tmp_path / "momo.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    spy_root = (repo_root / "data" / "raw").as_posix()

    cfg.write_text(f"""
run_name: momentum_streak_spy_smoke
strategy: momentum_streak
strategy_params:
  entry_streak: 3
  exit_streak: 2
  vol_lookback: 20
  vol_mult: 1.0
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
    assert (positions["qty"] < 0).any(), "expected at least one short position bar"
    assert "buy" in set(trades["side"]) and "sell" in set(trades["side"])
```

- [ ] **Step 2: Run the CLI integration tests**

Run: `python -m pytest tests/integration/test_run_backtest_cli.py -v`
Expected: pre-existing tests pass + 1 new test passes (3 total).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_run_backtest_cli.py
git commit -m "test(integration): CLI smoke test for momentum_streak on SPY"
```

---

### Task 8: WFO CLI smoke test on synthetic data

**Files:**
- Modify: `tests/integration/test_run_wfo_cli.py` (append one test)

- [ ] **Step 1: Append the test**

Append to `tests/integration/test_run_wfo_cli.py`:
```python
def test_run_wfo_cli_momentum_streak_emits_both_sides(tmp_path: Path):
    """WFO smoke test: stitched OOS trades must contain both BUY and SELL
    entries, proving the long/short momentum strategy ran end-to-end through
    the WFO orchestrator with allow_short=true."""
    raw = tmp_path / "data"
    raw.mkdir()
    # 900 bars is plenty for several WFO windows with train_bars=200.
    make_ohlcv(n=900, seed=23).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "wfo_momo.yaml"
    cfg.write_text(f"""
run_name: momentum_streak_wfo_smoke
strategy: momentum_streak
strategy_params:
  entry_streak: 3
  exit_streak: 2
  vol_lookback: 20
  vol_mult: 1.0
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
    entry_streak: [2, 3]
    exit_streak:  [1, 2]
    vol_lookback: [10, 20]
    vol_mult:     [1.0]
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
    sides = set(oos_trades["side"]) if len(oos_trades) else set()
    assert "buy" in sides, f"expected at least one BUY in oos_trades, got {sides}"
    assert "sell" in sides, f"expected at least one SELL in oos_trades, got {sides}"
```

**Note:** The 2×2×2×1 = 8-combo grid is intentionally tiny so this test finishes in under a minute. `test_run_wfo_cli.py` already imports `json`, `subprocess`, `sys`, `Path`, `pandas as pd`, and `make_ohlcv` (confirmed at the time this plan was written). No new imports are needed.

- [ ] **Step 2: Run the WFO integration tests**

Run: `python -m pytest tests/integration/test_run_wfo_cli.py -v`
Expected: pre-existing tests pass + 1 new test passes (3 total). Runtime: 30–60 seconds for the new test.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_run_wfo_cli.py
git commit -m "test(integration): WFO smoke test for momentum_streak"
```

---

## Phase 4: Final verification + docs touch-up

### Task 9: Full-suite green check

**Files:**
- (no file changes)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected:
- All 162 pre-existing tests pass (no edits).
- All new tests pass:
  - `test_strategy_momentum_streak.py` (8)
  - `test_run_backtest_cli.py` new test (1)
  - `test_run_wfo_cli.py` new test (1)
- Total: **172 passed**.

- [ ] **Step 2: If anything is red, stop and fix root cause.**

Likely failure modes to check:
- New test assertion drift (off-by-one in expected signal indices) — re-read the state-machine spec in Task 2.
- Registry test broke: shouldn't happen because `test_strategy_registry.py` does not enumerate IDs (verified during the short-positions work). If it did break, that means the strategy ID is being registered twice; check Task 3.
- WFO test exceeds 60s: increase the train_bars or shrink the grid further.

---

### Task 10: Update the README strategies list

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the strategies line in the Repository-layout section**

In `README.md`, find the line:

```
strategies/                # user/AI-authored strategies (sma_cross, rsi_mean_reversion, breakout_20d, rsi_long_short)
```

Replace with:

```
strategies/                # user/AI-authored strategies (sma_cross, rsi_mean_reversion, breakout_20d, rsi_long_short, momentum_streak)
```

- [ ] **Step 2: Bump the test count**

Find the line:

```
The test suite is **161 tests** covering every public surface ...
```

(NOTE: this said 161 at the README's last commit; after the WFO stitcher fix it became 162; this plan adds 10 more, so the new total is 172.)

Replace the number with `172` and the prose `all four sample strategies` with `all five sample strategies`:

```
The test suite is **172 tests** covering every public surface — types, exceptions, data loaders, validators, the engine (including signed-qty Position arithmetic and tri-state simulator transitions), analytics, all five sample strategies, the optimizer, WFO, and the four CLIs as end-to-end integration tests.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: list momentum_streak in README strategy lineup"
```

---

### Task 11: Optional — preview a backtest result

**Files:**
- (no file changes; manual verification only)

- [ ] **Step 1: Run all three workflows and inspect headline metrics**

Run the three sample configs in sequence:
```bash
python -m backtester.runners.run_backtest --config configs/backtests/momentum_streak_spy.yaml
python -m backtester.runners.run_optimize --config configs/optimize/momentum_streak_grid.yaml
python -m backtester.runners.run_wfo      --config configs/wfo/momentum_streak_wfo.yaml
```

Each command should exit 0 and log a one-line summary. There is no assertion to make here — this is a manual sanity check that all three CLIs end-to-end work on real SPY data.

- [ ] **Step 2: Done.** No commit needed for this manual verification step.

---

## Appendix A: Decision log

**Q: Vectorize the state machine or use an explicit loop?**
Explicit loop. The state at bar `i` depends on the state at bar `i-1` plus four conditions; a pure-vectorized solution would require either a sparse approach (ffill-based, like `breakout_20d` does) or numba, both more complex. A single O(n) Python loop over a few hundred to a few thousand bars is well below 10ms — not a bottleneck.

**Q: Exit on volume confirmation too?**
No. Exit on streak alone (per design spec §1.2). Rationale: exits should be responsive to price action even if volume is light, otherwise positions can be stuck in low-volume downturns.

**Q: Doji handling?**
Doji resets both streak counters. A doji is neither up nor down; in the `_consecutive_streak` helper, the mask is False for dojis in both `up` and `down`, so both counters return to 0. This matches the design spec §1.1.

**Q: Direct long→short flip semantics?**
The state machine checks `short_entry` before `long_exit` from the `+1` state. This means: if today's bar is a confirmed short-entry, we flip directly to -1 even if the exit_streak threshold has also been crossed. The v0.2.0 simulator emits one combined SELL order (closes the long and opens the short in one fill).

**Q: Why two new integration tests instead of three (one per workflow)?**
The WFO integration test exercises the grid-optimize path internally — `WalkForwardRunner` calls the optimizer per window. A standalone optimize-CLI test would duplicate coverage with no new failure mode. The two integration tests we ship (backtest + WFO) plus the eight unit tests provide enough end-to-end confidence.

## Appendix B: Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| All unit tests fail with `ModuleNotFoundError` | Strategy file not created (Task 2) or path wrong (`strategies/momentum_streak.py`, not `backtester/strategies/...`). |
| `test_streak_resets_on_doji` fails | `_consecutive_streak` not resetting on False bars; check `.where(mask, 0)` is present. |
| `test_long_entry_suppressed_when_volume_below_threshold` fails | Volume confirmation comparing against NaN SMA returning NaN→True instead of False; check `.fillna(False)` after the comparison. |
| `test_direct_long_to_short_flip` returns 0 at idx 10 instead of -1 | State machine in Task 2 checks `lx` before `se` from the `+1` state; the spec requires `se` first (flip wins over plain exit). |
| Backtest CLI run times out or hangs | Likely a bug in the strategy that produces infinite signals or an exception during simulation. Run with `python -X dev` and check `logs.txt` for stack traces. |
| Registry one-liner shows `momentum_streak` missing | Forgot to add the `register_strategy(MomentumStreakStrategy)` call in Task 3. The import alone isn't enough. |
| WFO smoke test fails with `ShortNotAllowedError` | Config forgot `allow_short: true`. Check the YAML fixture in Task 8. |
