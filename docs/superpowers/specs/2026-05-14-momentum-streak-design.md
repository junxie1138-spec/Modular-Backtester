# `momentum_streak` strategy — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Add a basic momentum strategy to the Modular Backtester that takes the long side after a configurable run of consecutive up-days confirmed by above-average volume, and the short side symmetrically after a confirmed run of down-days. Wire it through all three workflows: single backtest, grid optimization, and walk-forward optimization (WFO).

**Version target:** Drop into the existing `v0.2.x` line — no engine, simulator, or framework changes required. The strategy is one new module plus sample configs and tests.

---

## 1. Behavior

### 1.1 Definitions
- **Green day**: `close[i] > close[i-1]`.
- **Red day**: `close[i] < close[i-1]`.
- **Doji / flat day**: `close[i] == close[i-1]`. Counts as neither green nor red and **resets both streak counters to 0**.
- **Green streak at bar `i`**: the number of consecutive green days ending at bar `i` (inclusive). Resets to 0 on the first non-green bar.
- **Red streak at bar `i`**: symmetric.
- **Volume confirmation at bar `i`**: `volume[i] > vol_mult * rolling_mean(volume, vol_lookback)[i]`. The rolling mean is computed over the trailing `vol_lookback` bars including bar `i` (standard pandas `.rolling(vol_lookback).mean()` semantics).

### 1.2 Signal generation

The strategy maintains a position state `state ∈ {-1, 0, +1}`. At each bar `i ≥ 1`, the following conditions are evaluated:

- `long_entry  = green_streak[i] >= entry_streak  AND vol_confirm[i]`
- `short_entry = red_streak[i]   >= entry_streak  AND vol_confirm[i]`
- `long_exit   = red_streak[i]   >= exit_streak`
- `short_exit  = green_streak[i] >= exit_streak`

Exits do **not** require volume confirmation. The state machine transitions are:

```
From flat (state = 0):
  if long_entry:  state := +1
  elif short_entry: state := -1
  else: state remains 0

From long (state = +1):
  if short_entry: state := -1   # direct flip via combined order (handled by simulator)
  elif long_exit:  state := 0
  else: state remains +1

From short (state = -1):
  if long_entry:  state := +1   # direct flip
  elif short_exit: state := 0
  else: state remains -1
```

When `entry_streak >= 1` and `exit_streak >= 1` the two simultaneous events (e.g., long_entry and long_exit on the same bar — only possible with a doji intervening, which we already exclude) are mutually exclusive by construction.

### 1.3 Signal-frame contract

The strategy emits a `SignalFrame` with:
- `signal` column: `state.shift(1).fillna(0).astype(int)` — the standard one-bar shift so the simulator fills on the next bar's open, matching every other strategy in this project.
- `size` column: a constant copy of `params.size`.
- No `price_column` (MARKET orders only — this strategy does not use LIMIT entries).

The first bar's signal is always 0 after the shift.

### 1.4 Warmup

`warmup_bars = max(entry_streak, exit_streak, vol_lookback) + 1`. The `+1` accounts for the `close.diff()` operation needed to detect green/red days.

### 1.5 Requirement for shorts

Emitting `-1` requires `execution.allow_short: true` in the run config. Otherwise the simulator raises `ShortNotAllowedError` on the first short entry. The strategy's docstring will state this requirement (mirroring `rsi_long_short`).

---

## 2. Parameters

`MomentumStreakParams` (`@dataclass(slots=True)`):

| Name | Type | Default | Notes |
|---|---|---|---|
| `entry_streak` | `int` | `3` | Consecutive same-direction bars required to trigger entry. Must be `>= 1`. |
| `exit_streak` | `int` | `2` | Consecutive opposite-direction bars to flatten an existing position. Must be `>= 1`. |
| `vol_lookback` | `int` | `20` | Rolling window for the volume SMA. Must be `>= 2`. |
| `vol_mult` | `float` | `1.0` | Confirmation multiplier on the entry bar. `1.0` = above average; `1.5` = 50% above average; etc. Must be `> 0`. |
| `size` | `float` | `1.0` | Position-size multiplier (applies to both legs of a flip). Project convention. |

Parameter validation is not enforced inside the strategy — the project does not validate strategy params at the dataclass layer; YAML loader uses `**raw`. Out-of-range values produce empty or all-zero signals (acceptable degenerate behavior).

---

## 3. File layout

| File | Action |
|---|---|
| `strategies/momentum_streak.py` | create |
| `backtester/strategies/registry.py` | modify (one import + one `register_strategy(...)` line) |
| `configs/backtests/momentum_streak_spy.yaml` | create |
| `configs/optimize/momentum_streak_grid.yaml` | create |
| `configs/wfo/momentum_streak_wfo.yaml` | create |
| `tests/unit/test_strategy_momentum_streak.py` | create |
| `tests/integration/test_run_backtest_cli.py` | append one test |
| `tests/integration/test_run_wfo_cli.py` | append one test |

No changes to engine, broker, portfolio simulator, position, fills, optimizer, splitter, stitcher, runners, or analytics.

---

## 4. Implementation notes

### 4.1 Indicators

The `indicators()` method returns a DataFrame with these columns (all aligned to `data.index`):

- `up: bool` — `data["close"] > data["close"].shift(1)`
- `down: bool` — `data["close"] < data["close"].shift(1)`
- `green_streak: int` — see vectorized recipe below
- `red_streak: int` — symmetric
- `vol_sma: float` — `data["volume"].rolling(vol_lookback).mean()`
- `vol_confirm: bool` — `data["volume"] > vol_mult * vol_sma` (NaN-safe: where `vol_sma` is NaN during warmup, `vol_confirm` evaluates to `False`)

**Vectorized streak counting**: pandas idiom — group consecutive True runs.

```python
def _streak(mask: pd.Series) -> pd.Series:
    # cumcount within consecutive runs of True; 0 on False bars.
    grp = (mask != mask.shift(1)).cumsum()
    return mask.groupby(grp).cumcount().add(1).where(mask, 0).astype(int)
```

This avoids a Python-level for-loop over bars. The first NaN-from-shift cell is handled by `.fillna(False)` on `mask` before the groupby.

### 4.2 State machine

The state machine in `generate_signals()` is the only loop in the strategy and runs once over the index. It is O(n) and operates on already-vectorized booleans:

```python
state = np.zeros(len(data), dtype=int)
for i in range(1, len(data)):
    prev = state[i - 1]
    le = long_entry[i]; se = short_entry[i]
    lx = long_exit[i];  sx = short_exit[i]
    if prev == 0:
        state[i] = 1 if le else (-1 if se else 0)
    elif prev == 1:
        state[i] = -1 if se else (0 if lx else 1)
    else:  # prev == -1
        state[i] = 1 if le else (0 if sx else -1)
```

Then `signal = pd.Series(state, index=data.index).shift(1).fillna(0).astype(int)`.

---

## 5. Tests

### 5.1 Unit (`tests/unit/test_strategy_momentum_streak.py`)

Eight tests:

1. **Green streak counts correctly** — synthetic series of `+1, +1, +1, -1, +1` produces streaks `1, 2, 3, 0, 1`.
2. **Streak resets on doji** — series of `+1, 0, +1` produces green streaks `1, 0, 1` and red streaks `0, 0, 0`.
3. **Long entry fires on streak + volume** — `entry_streak=3`, volume on the 3rd green bar > `vol_mult * SMA` → signal becomes `+1` on the bar after.
4. **Long entry suppressed when volume below threshold** — same setup, volume below threshold → signal stays `0`.
5. **Long exit fires after `exit_streak` reds** — long position, then `exit_streak` consecutive reds → signal becomes `0`.
6. **Short entry symmetric to long** — `entry_streak` reds + volume confirms → signal becomes `-1`.
7. **Direct long→short flip on opposite high-volume streak** — sequence engineered so long is open, then `entry_streak` reds with high volume → signal goes from `+1` directly to `-1` (no intervening `0`).
8. **Signal is shifted one bar; warmup formula is correct** — first bar's signal is `0`; `strat.warmup_bars(params)` equals `max(entry_streak, exit_streak, vol_lookback) + 1`.

### 5.2 Integration

- **CLI backtest smoke** (append to `tests/integration/test_run_backtest_cli.py`): run the new strategy via the CLI on bundled SPY data with `allow_short: true`. Assert `summary["symbol"] == "SPY"`, `summary["n_trades"] > 0`, `trades.csv` contains both `buy` and `sell`, `positions.csv` has at least one negative `qty`.
- **CLI WFO smoke** (append to `tests/integration/test_run_wfo_cli.py`): run WFO with a small grid (2×2×2 = 8 combos) over 900-bar synthetic data. Assert stitched `oos_trades.csv` contains both `buy` and `sell` sides.
- Grid-optimize is implicitly exercised by the WFO test (WFO runs the optimizer per window). A standalone grid-optimize CLI test is not added — it would be duplicative.

---

## 6. Config samples

### 6.1 `configs/backtests/momentum_streak_spy.yaml`

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

### 6.2 `configs/optimize/momentum_streak_grid.yaml`

Same as backtest config, with `optimization` block added:

```yaml
optimization:
  objective: sharpe
  param_space:
    entry_streak: [2, 3, 5]
    exit_streak:  [1, 2, 3]
    vol_lookback: [10, 20, 50]
    vol_mult:     [1.0, 1.25, 1.5]
```

→ 3×3×3×3 = 81 combinations. Runtime budget: well under a minute on a 10-year SPY dataset.

### 6.3 `configs/wfo/momentum_streak_wfo.yaml`

Same as the grid config, plus:

```yaml
wfo:
  enabled: true
  train_bars: 756   # 3 trading years
  test_bars: 252    # 1 trading year
  step_bars: 252
```

---

## 7. Out of scope

The following are deliberately not part of this design and should be considered separate, follow-up features if needed:

- ATR-based or volatility-scaled position sizing.
- A "minimum streak gap" between consecutive entries.
- Asymmetric long/short parameters (e.g., longer streaks for shorts).
- Volume z-score or other smoother confirmation (the score-based approach considered as "Option C" in brainstorming).
- LIMIT-order entries on the next bar.

---

## 8. Acceptance criteria

1. `python -m pytest -q` passes with all existing tests still green plus the new ones (target: previous total + 10 new = **172 tests**).
2. `python -m backtester.runners.run_backtest --config configs/backtests/momentum_streak_spy.yaml` exits 0 and writes the standard artifact bundle.
3. `python -m backtester.runners.run_optimize --config configs/optimize/momentum_streak_grid.yaml` exits 0 and writes `grid_results.json` with 81 rows.
4. `python -m backtester.runners.run_wfo --config configs/wfo/momentum_streak_wfo.yaml` exits 0, writes `window_results.json`, and stitched OOS trades contain both `buy` and `sell` rows.
5. `python -c "from backtester.strategies.registry import STRATEGY_REGISTRY; print(sorted(STRATEGY_REGISTRY))"` lists `momentum_streak` alongside the existing four.
6. No regressions in any pre-existing test or sample run.
