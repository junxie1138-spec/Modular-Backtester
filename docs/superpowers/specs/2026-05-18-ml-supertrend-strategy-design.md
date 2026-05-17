# ml_supertrend Strategy — Design

**Date:** 2026-05-18
**Status:** Approved, ready for implementation plan

## 1. Purpose

Port the signal core of the TradingView Pine Script *"Machine Learning Supertrend
[Aslan]"* into a curated strategy module that runs in this backtester.

The Pine Script is two layers bolted together:

1. A **signal core** — a SuperTrend (ATR band) trend filter combined with
   new-high/new-low detection, gated by an RSI filter, an optional volume-surge
   filter, an optional "key levels" filter, and a minimum-spacing rule. Two
   modes: Reversal and Breakout.
2. An **adaptive "ML" engine** (Pine input groups ⑥–⑫) — a self-tuning loop
   that runs phantom probe trades, scores them, and continuously nudges the
   SuperTrend parameters via an optimizer, a regime grid, and decay traces.

**Only the signal core is ported.** The adaptive engine is intentionally
dropped, for two reasons:

- It tunes parameters on the same data it trades — in-sample fitting that
  inflates backtest numbers and is not a sound research method.
- This suite already solves "find good parameters" properly, with grid-search
  and walk-forward optimization. Those replace the Pine self-tuner.

The strategy keeps the recognizable name `ml_supertrend`, but its docstring
states plainly that the adaptive engine was removed and that parameters are
static (tune them with the suite's optimizers).

## 2. Scope

### In scope

- One new file `strategies/ml_supertrend.py`.
- One registration line in `backtester/strategies/registry.py`.
- One backtest config `configs/backtests/ml_supertrend_spy.yaml`.
- One unit test `tests/unit/test_strategy_ml_supertrend.py`.

### Out of scope (dropped from the Pine Script)

- The entire adaptive engine: Master Dial, Auto-Tune, Optimizer, Risk Guard,
  Context Memory regime grid, decay traces, micro-batch processing, the
  background test matrix (Pine groups ⑥–⑫).
- Realtime-only features: live tick-pressure sensor, state snapshot
  serialize/restore, execution ledger.
- The Pine "Risk Management" visual TP/SL block. Exits are handled by the
  position model (§5); the suite's config-level stops remain available but are
  not wired on by default.

## 3. Interface

Standard vectorized strategy — `BaseStrategy` subclass with `indicators()` and
`generate_signals()`. **Not** per-bar mode: every input is derivable from OHLCV
alone, with no dependence on simulator state (position phase, regime, fills).

```
strategy_id = "ml_supertrend"
version     = "1.0"
asset_type  = "stock"
timeframe   = "1d"
```

`warmup_bars` = `max(atr_period, sensitivity, rsi_len, vol_lookback) + 1`.

## 4. Parameters

`MLSupertrendParams` — a `@dataclass(slots=True)` mirroring Pine input
groups ①–⑤ plus the relevant ⑬ fields. Defaults are copied from the Pine
`input(...)` defaults.

| Field | Type | Default | Pine origin |
|---|---|---|---|
| `signal_mode` | str | `"reversal"` | ① Signal Type — `"reversal"` or `"breakout"` |
| `require_new_extreme` | bool | `True` | ① Require Fresh Pivot |
| `min_bars_between_signals` | int | `10` | ① Signal Spacing |
| `sensitivity` | int | `30` | ② Lookback Window |
| `atr_period` | int | `24` | ② Smoothing Period |
| `multiplier` | float | `1.4` | ② Band Width |
| `source_type` | str | `"hlcc4"` | ② Price Basis |
| `use_atr` | bool | `True` | ② True Range Mode (RMA vs EMA smoothing) |
| `enable_rsi` | bool | `True` | ③ Active |
| `rsi_len` | int | `14` | ③ Length |
| `rsi_lookback_top` | int | `50` | ③ Hot Zone Memory |
| `rsi_lookback_bot` | int | `50` | ③ Cold Zone Memory |
| `rsi_top` | int | `70` | ⑬ RSI Hot Level |
| `rsi_bot` | int | `30` | ⑬ RSI Cold Level |
| `vol_lookback` | int | `3` | ④ Sample Depth |
| `vol_multiplier` | float | `1.2` | ④ Surge Threshold |
| `require_vol_spike` | bool | `False` | ④ Require Surge |
| `enable_major_levels_only` | bool | `False` | ⑤ Key Levels Only |
| `major_level_threshold` | float | `4.5` | ⑤ Key Level Depth (xATR) |
| `size` | float | `1.0` | position size weight |

Unmapped: any Pine input belonging to groups ⑥–⑫ (adaptive engine) or the
realtime/snapshot features is dropped, not represented as a parameter.

`source_type` resolves to a price series: `open/high/low/close` direct;
`hl2 = (h+l)/2`; `hlc3 = (h+l+c)/3`; `ohlc4 = (o+h+l+c)/4`;
`hlcc4 = (h+l+c+c)/4`.

`size` is a **constant scalar**, not varied per bar — the strategy emits one
fixed value into the `size` column for every bar (matching `rsi_long_short.py`).
The engine reads it as a per-signal weight and multiplies it into the
config-level position-sizing rules (`portfolio.sizing_mode` / `portfolio.size`).
There is no regime-based or adaptive sizing — that would be part of the dropped
adaptive engine. The field is kept so a future variant could vary it, but for
this strategy it is always `params.size`.

## 5. Position model — stop-and-reverse (long/short)

The Pine `indicator()` only paints Buy/Sell arrows; it never defines an exit.
We map the Buy/Sell event stream to an always-in-the-market position:

- A **Buy** event → hold `signal = +1` until the next Sell.
- A **Sell** event → flip to `signal = -1` until the next Buy.
- The position carries forward bar to bar; it never sits flat once the first
  signal has fired.

This requires `execution.allow_short: true` in the config. With it absent the
portfolio simulator raises `ShortNotAllowedError` on the first `-1`.

**Stop-and-reverse is realized entirely strategy-side.** The engine has no
"position model" config knob — it simply treats the `signal` series as the
target position each bar ({-1, 0, +1}). The always-in-the-market behaviour is
produced by `generate_signals()` emitting a never-flat held series; the *only*
config-layer requirement is `allow_short: true`. There is therefore no
`position_model:` key to set in the YAML.

## 6. Computation

### 6.1 `indicators()` — vectorized

| Column | Definition |
|---|---|
| `atr` | True Range smoothed: RMA (`alpha = 1/atr_period`) when `use_atr`, else EMA (`alpha = 2/(atr_period+1)`). The suite's `compute_atr` uses SMA smoothing and is **not** used — Pine's signal path uses RMA/EMA, so smoothing is computed inline to stay faithful. |
| `st_trend` | SuperTrend trend state in {+1, -1}. Bands and the trend flip are a sequential recurrence (see §6.2) — computed with a single numpy loop. |
| `rsi` | Wilder RSI over `rsi_len`. RSI uses **Wilder smoothing** (`alpha = 1 / rsi_len`), matching the Pine `ta.rsi` — Wilder smoothing as in `rsi_long_short.py`, not a generic library RSI. One refinement over `rsi_long_short.py`: a window with no down moves yields `RSI = 100` (the Pine convention) instead of `NaN`, so the RSI filter is not silently disabled during a strong rally. |
| `roll_high` / `roll_low` | Rolling max(high) / min(low) over `sensitivity` bars. |
| `is_new_high` / `is_new_low` | A fresh extreme: `roll_high` differs from its value `lookback` bars ago **and** `close` exceeds that prior value. `lookback = max(1, round(sensitivity / 10))`. Mirror for lows. |
| `rsi_cold` / `rsi_hot` | `rsi < rsi_bot` (resp. `> rsi_top`) was true on at least one of the last `rsi_lookback_bot` (resp. `_top`) bars — a rolling-window "any" over the boolean. When `enable_rsi` is false both are constant `True`. |
| `vol_surge` | `volume > vol_multiplier * SMA(volume, vol_lookback)`. |
| `sig_high` / `sig_low` | `sig_high = is_new_high`, and when `enable_major_levels_only` also `high - roll_low > atr * major_level_threshold`. `sig_low = is_new_low`, and when the filter is on also `roll_high - low > atr * major_level_threshold`. When the filter is off, `sig_high == is_new_high` and `sig_low == is_new_low`. |

**Major-levels filter — exact formula.** The depth is measured from the *same*
`roll_high` / `roll_low` windows used for extreme detection (rolling max(high) /
min(low) over `sensitivity` bars) — no separate baseline. The `atr` term is the
**current bar's** smoothed ATR column, not a lagged ATR; using the current bar
is safe because look-ahead is prevented by the single `shift(1)` applied to the
final signal series (§6.4), not by lagging individual indicators.

### 6.2 SuperTrend recurrence (faithful port)

Per the Pine `getSupertrend_var`, with `src` from `source_type`:

```
support    = src - multiplier * atr      # band below price
resistance = src + multiplier * atr      # band above price
support    = close[-1] > support[-1]    ? max(support, support[-1])       : support
resistance = close[-1] < resistance[-1] ? min(resistance, resistance[-1]) : resistance
trend = -1 and close > resistance[-1] -> +1
trend = +1 and close < support[-1]    -> -1
trend starts at +1
```

(The Pine names these `upper`/`lower` with swapped meanings; the design uses
`support`/`resistance` for clarity. Behaviour is identical.)

### 6.3 `generate_signals()` — one sequential pass

Three things are stateful and cannot be expressed as pure column math:
minimum signal spacing, the Reversal-mode `topFlag`/`botFlag` latches, and
carrying the stop-and-reverse position. A single numpy loop over bars handles
all three.

**Reversal mode** (`signal_mode == "reversal"`):
- `top_flag` latches to 1 when `sig_high` occurs during `st_trend == +1`;
  resets to 0 when `st_trend == -1`. `bot_flag` is the mirror.
- A **Sell** fires when `top_flag` falls 1→0, i.e. a fresh high was registered
  and the trend has now flipped down. **Buy** is the mirror.
- When `require_new_extreme` is false, a bare SuperTrend flip
  (`+1→-1` Sell, `-1→+1` Buy) also fires.

**Breakout mode** (`signal_mode == "breakout"`):
- **Sell** when `sig_high and st_trend == +1`; **Buy** when
  `sig_low and st_trend == -1`. (Faithful to the Pine: a new high triggers a
  Sell — this is a contrarian design.)

In both modes a candidate signal is suppressed unless
`bars_since_last_signal >= min_bars_between_signals`. A Buy additionally
requires `rsi_cold` (and `vol_surge` when `require_vol_spike`); a Sell requires
`rsi_hot` (and `vol_surge` when `require_vol_spike`).

The loop emits the buy/sell event stream, then carries the held position
per §5. The result is `shift(1)`-ed (enter on the bar after the signal) and
returned as a `SignalFrame` with `signal` ∈ {-1, 0, 1} and `size`.

### 6.4 Warmup and look-ahead

- `warmup_bars = max(atr_period, sensitivity, rsi_len, vol_lookback) + 1`.
- For bars before `warmup_bars`, `generate_signals()` emits `signal = 0` and
  ignores any candidate events. The first non-zero signal may only appear once
  every indicator column in §6.1 is valid (non-NaN) for that bar.
- Look-ahead is prevented by a **single** `shift(1)` on the assembled position
  series — the entry lands on the bar *after* the signal bar. Individual
  indicators are computed on the current bar and are not separately lagged.

## 7. Deliverables and registration

- `strategies/ml_supertrend.py` — `MLSupertrendParams` + `MLSupertrendStrategy`.
- `backtester/strategies/registry.py` — add the import and
  `register_strategy(MLSupertrendStrategy)` line in the curated block.
- `configs/backtests/ml_supertrend_spy.yaml` — SPY, `1d`, modelled on
  `sma_cross_spy.yaml`, with `execution.allow_short: true`. Its
  `strategy_params` block uses the **Pine-default values verbatim** (the §4
  defaults) and a header comment labels it the *"TradingView-parity baseline"*.
  Any future tuned config (grid-search / WFO output) is named distinctly so a
  reader can always tell a TV-parity run from an optimized one.
- `tests/unit/test_strategy_ml_supertrend.py` — modelled on
  `test_strategy_rsi_long_short.py`.

## 8. Testing strategy

Unit test coverage:

- `params_type()` returns `MLSupertrendParams`; defaults match the table in §4.
- `indicators()` produces every column in §6.1 with no look-ahead (warmup
  region is NaN/neutral).
- SuperTrend `st_trend` only ever holds {+1, -1} and flips at the documented
  crossings on a hand-built series.
- `generate_signals()` output `signal` ∈ {-1, 0, 1}; the stop-and-reverse
  position alternates correctly across a constructed Buy/Sell sequence.
- `min_bars_between_signals` is honoured — no two signals closer than the gap.
- The first non-warmup signal is `shift(1)`-ed (no entry on the signal bar).

Integration: the strategy runs end-to-end via
`python -m backtester.runners.run_backtest --config configs/backtests/ml_supertrend_spy.yaml`
and produces a normal artifact bundle.

### 8.1 Illustrative case — reversal vs breakout

The mode-difference test builds one hand-crafted OHLCV series and runs it
through both modes. Construct a sequence where price climbs to a fresh high
(`sig_high` true) while `st_trend == +1`, then rolls over so `st_trend` flips
`+1 → -1` a few bars later:

- **Breakout mode** fires a **Sell on the fresh-high bar itself**
  (`sig_high and st_trend == +1`).
- **Reversal mode** fires **nothing on that bar** — it only latches
  `top_flag = 1`. The Sell fires **later**, on the bar where `st_trend` flips
  `+1 → -1` (`top_flag` falls `1 → 0`).

So on the *same* series Breakout enters earlier (at the extreme) and Reversal
enters later (at the confirmed flip). The test asserts the Sell signal bar
index differs between the two modes by exactly the gap between the fresh-high
bar and the trend-flip bar. (Stated qualitatively in trend-state terms; the
test fixture pins the exact bar indices.)

## 9. Risks and notes

- **Contrarian polarity.** A new *high* yields a *Sell*. This is faithful to the
  Pine Script (its own UI calls the signals "Contrarian"). Not a bug — the test
  asserts this polarity so it is not silently "corrected" later.
- **SuperTrend is recursive.** The numpy loop in `indicators()` is unavoidable;
  `_ols_slope` in `mean_reversion_atr.py` sets the precedent that per-bar loops
  inside `indicators()` are acceptable in this codebase.
- **Stop-and-reverse needs shorting.** The shipped config sets
  `allow_short: true`; the strategy docstring states this requirement so a user
  who copies the strategy into a long-only config gets a clear failure.
