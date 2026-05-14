# `mean_reversion_atr` strategy + v0.4.0 framework — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Add a defense-first swing-trading mean-reversion strategy (`mean_reversion_atr`) to the Modular Stock Backtester. Entry is `close ≤ mean10 − 1.25·ATR20`. Exit is asymmetric: tranche 1 (50% of position) exits at the 10-day mean; the runner (50%) trails at `2.5·ATR` below the highest *close* since entry with a breakeven floor, and is capped by a hard ceiling of `mean10 + 1.25·ATR`. Position sizing is volatility-targeted, capped per name and per sector, with a cross-symbol risk budget. Three independent regime gates (SPY 200-EMA, VIX hysteresis, rolling 20-day strategy PnL circuit breaker) trip the book to full cash. Universe is a manually vetted 15-name list (TSLA, NVDA, AMD, COIN, GOOGL, MSTR, XPEV, NIO, PLTR, SMCI, SHOP, W, META, NFLX, + 1 TBD) plus auxiliary OHLCV inputs for SPY and ^VIX.

**Version target:** New framework cycle. Bumps `pyproject.toml` from `0.3.0` to `0.4.0`. This is a framework version bump, not a feature addition — the simulator becomes multi-symbol, the strategy contract gains an auxiliary-data slot and per-bar callbacks, and the optimizer gains a Latin-hypercube sampling mode. The v0.3.0 single-symbol code path remains intact for backwards compatibility.

**Scope.** This spec resolves Slice C of the brainstorming scope ladder. The PRD spans both the strategy layer and the framework layer, and the brainstorm explicitly chose the full framework rewrite over the strategy-only or strategy-plus-aux-data slices. Slice A and Slice B are not in scope here.

---

## 0. Architectural reality check

### 0.1 Pipeline after v0.4.0

```
load_run_config + load_universe_config
        │
        ▼
yfinance_loader.load_panel(universe + aux symbols)     ← new
        │
        ▼
MultiSymbolBacktestEngine                              ← new, alongside v0.3.0 BacktestEngine
        │
        ▼
MultiSymbolPortfolioSimulator                          ← new
   ├── shared cash + equity
   ├── per-symbol TrancheStopState                     ← new (separate from v0.3.0 TrailingStopState)
   ├── RegimePolicy (SPY/VIX/circuit breaker)          ← new
   ├── RiskBudgetEnforcer (sum pos × stop_dist ≤ 6%)   ← new
   ├── SectorCapEnforcer (≤ 50% deployed per sector)   ← new
   └── per-symbol Broker + FillEngine + Position       ← reused from v0.3.0
        │
        ▼
ArtifactWriter (extended for multi-symbol artifacts)
```

v0.3.0's `BacktestEngine` + `PortfolioSimulator` + `TrailingStopState` stay untouched. Strategies that emit `signal ∈ {-1, 0, 1}` and don't read auxiliary data continue to use the v0.3.0 path. The v0.4.0 path is opt-in via `strategy.uses_multi_symbol = True` (a class attribute, defaulting to False).

### 0.2 Strategy contract additions (v0.4.0 only)

| Addition | Mechanism |
|---|---|
| Fractional target positions | `SignalFrame.signal` accepts any float in `[-1.0, 1.0]`, not just `{-1, 0, 1}`. Half-size = `0.5`. |
| Auxiliary OHLCV data | `generate_signals` and `indicators` receive `aux_data: dict[str, pd.DataFrame]` keyed by aux symbol. |
| Per-bar callback mode | Strategies set `class.uses_per_bar = True` to be called once per bar instead of once over the full timeline. Required for mean_reversion_atr because tranche-phase logic depends on simulator-driven state. |
| StrategyContext extensions | `ctx.position_phase: dict[str, TSPhase]`, `ctx.bars_in_phase: dict[str, int]`, `ctx.recent_pnl: pd.Series`, `ctx.regime: RegimeState`. All read-only from the strategy's perspective. |
| Multi-symbol declaration | `strategy.uses_multi_symbol = True` opts into the v0.4.0 code path. Default False keeps v0.3.0 strategies untouched. |

A strategy that does not set these attributes continues to work unchanged.

### 0.3 Goals and non-goals

The PRD specifies performance targets — ~20% annualized return, max 10% drawdown, Calmar ≈ 2.2 — and stress-window kill criteria. These are **strategy-tuning targets**, not framework-correctness gates. v0.4.0 ships when the framework works correctly. Whether the strategy hits the PRD's performance numbers is a separate question, tunable within the v0.4.0 framework by adjusting `mean_rev_v04.yaml` and `universe.yaml`. See §10 for the precise wording of acceptance gates and the xfail-by-default machinery that distinguishes framework regressions from tuning gaps.

---

## 1. Strategy behavior

### 1.1 Definitions

- `close[i]` — adjusted close on bar `i` (see §5.1 for the framework's adjustment contract).
- `mean10[i] = SMA(close, params.mean_lookback)[i]`. Default `mean_lookback = 10`.
- `ATR20[i] = compute_atr(data, params.atr_lookback)[i]` using `backtester/engine/atr.py` (true-range, SMA-smoothed). Default `atr_lookback = 20`.
- `slope_log[i]` — rolling-200 OLS slope of `ln(close)` on bar index. Used by the runtime trend gate.
- `trend_active[i] = abs(expm1(slope_log[i])) > params.runtime_trend_threshold`. Default `runtime_trend_threshold = 0.0025` (≈ 0.25% per day).
- A position's **phase** is `TSPhase ∈ {HARD, RUNNER, DISARMED}`, owned by `TrancheStopState` and exposed to the strategy through `ctx.position_phase[symbol]`.

### 1.2 Entry rule

A long entry fires on bar `i` if **all** of the following hold:

1. `ctx.position_phase[symbol] == DISARMED` (no open position).
2. `close[i] ≤ mean10[i] − params.entry_atr_mult * ATR20[i]`. Default `entry_atr_mult = 1.25`.
3. `not trend_active[i]` (runtime trend gate; pre-screened symbols rarely cross this).
4. `not ctx.regime.book_flat` (no regime gate currently tripped — see §4).
5. The simulator can fit the new position within all of: `position_cap_pct`, `cash_reserve_pct`, `risk_budget_pct`, `sector_cap_pct`. The strategy emits `target = 1.0`; the simulator scales/drops as needed.

Short entries are **not** part of this strategy. `mean_reversion_atr` is long-only and does not set `execution.allow_short`.

### 1.3 Exit rules

Once a position is open, the strategy emits the target sequence below. The simulator translates each target into the appropriate share delta.

| Phase | Trigger | Target emitted | Effect |
|---|---|---|---|
| HARD | `close[i] ≥ mean10[i]` | `0.5` | Tranche 1 fills; simulator detects qty drop and promotes the position to RUNNER. |
| HARD | `i − entry_bar ≥ params.time_stop_days` (default 7) | `0` | Full-position time stop. Both tranches exit. |
| HARD | TrancheStopState's hard stop fires (`low ≤ entry_price − params.hard_stop_atr_mult * atr_at_entry`) | n/a — execution layer | Position closed by the simulator. |
| RUNNER | `close[i] ≥ mean10[i] + params.runner_ceiling_atr_mult * ATR20[i]` | `0` | Hard ceiling: "just take it" exit prevents the runner from riding a name into a range breakdown. |
| RUNNER | `i − tranche_1_fill_bar ≥ params.runner_time_stop_days` (default 12) | `0` | Runner time stop. |
| RUNNER | TrancheStopState's runner trail fires (`low ≤ max(entry_price, peak_close − params.runner_atr_mult * ATR20[i])`) | n/a — execution layer | Position closed by the simulator. |
| Any | `ctx.regime.book_flat` becomes True (any gate trips) | `0` | All positions flattened; new entries suppressed until all gates reset. |

The runner ceiling and runner time stop ONLY apply once the position is in RUNNER phase. The 7-day full-position time stop only applies in HARD phase. The PRD's wording is enforced by reading `ctx.position_phase[symbol]` in `generate_signals`.

### 1.4 Tranche-1 mean-drift acknowledgement (PRD open item 4)

The 10-day mean drifts down while a long position is held. Tranche 1's mean-reversion exit therefore exits at a price that may be *below* the entry, even when the strategy thesis is correct. This is an accepted feature of v0.4.0: the runner's 2.5×ATR trail (with breakeven floor) offsets it at the position level, but the half-position drift is not formally resolved in this spec. The brainstorming round accepted this trade-off after considering the alternative ("exit tranche 1 at a higher level than the live mean") and rejecting it as over-fitting.

### 1.5 Re-entry after a stop or signal-driven exit

There is no cooldown. If the position closes on bar `i` (stop or signal) and the entry rule fires again on bar `i+1`, the simulator opens a new position. This matches v0.3.0's no-cooldown semantics. The PRD's discussion of "free risk budget on tranche 1 fill" implies new entries can use the freed budget on the same bar; the simulator's risk-budget enforcer (§3.3) honors that.

### 1.6 `position_phase` finalization

`ctx.position_phase[symbol]` reflects the phase at the **close of the just-processed bar**. The strategy's decision on bar `t+1` is based on finalized state from bar `t`. A same-bar partial close that happens at bar `t`'s open (from an order scheduled at bar `t−1`) is therefore visible to the strategy when it computes its bar-`t+1` signal, not on the next-but-one signal computation. Implementation: the per-bar strategy callback runs after step 5 (peak/trough update) and before step 7 (schedule next-bar orders) in the simulator's loop (§3.1).

---

## 2. TrancheStopState (v0.4.0)

### 2.1 Position vs. v0.3.0 TrailingStopState

v0.3.0's `TrailingStopState` remains untouched. It tracks intrabar `peak_high`/`trough_low`, has a single armed state, and is driven by `execution.trailing_stop_pct` or `execution.trailing_stop_atr_mult`. v0.4.0 introduces a separate class `TrancheStopState` in `backtester/engine/tranche_stop.py`.

Mutual exclusion: a run YAML may use the v0.3.0 keys OR the v0.4.0 keys (`hard_stop_atr_mult` + `runner_atr_mult` + `breakeven_floor`), never both. Validation rejects the combination.

### 2.2 Three phases

```python
class TSPhase(Enum):
    HARD = "hard"          # entry → tranche 1 fill: fixed hard stop, no trailing
    RUNNER = "runner"      # tranche 1 fill → exit: close-basis trail with breakeven floor
    DISARMED = "disarmed"  # flat
```

### 2.3 Fields

```python
@dataclass
class TrancheStopState:
    # configuration (immutable per run)
    hard_stop_atr_mult: float                  # PRD default 1.75
    runner_atr_mult:    float                  # PRD default 2.5
    breakeven_floor:    bool = True
    atr_series:         pd.Series              # aligned to data.index; same source as v0.3.0 atr.py

    # snapshotted at reset() — frozen for the life of the position
    entry_price:        float = 0.0
    entry_bar_idx:      int = -1
    atr_at_entry:       float = float("nan")

    # mutating state — tracks the position
    phase:              TSPhase = TSPhase.DISARMED
    peak_close:         float = 0.0
    trough_close:       float = float("inf")
```

Note: there are no `peak_high` or `trough_low` fields on `TrancheStopState`. Intrabar wicks do not move the runner trail. Only confirmed closes do. The field names are chosen to make this contract obvious.

### 2.4 Methods

```python
def reset(self, entry_price: float, bar_idx: int) -> None:
    """Called on flat → non-flat transition."""
    self.entry_price   = entry_price
    self.entry_bar_idx = bar_idx
    self.atr_at_entry  = float(self.atr_series.iloc[bar_idx])
    self.peak_close    = entry_price
    self.trough_close  = entry_price
    self.phase         = TSPhase.HARD

def promote_to_runner(self) -> None:
    """Called by the simulator on detected partial close (same-sign qty drop)."""
    if self.phase == TSPhase.HARD:
        self.phase = TSPhase.RUNNER
    # idempotent: no-op if already RUNNER or DISARMED.

def disarm(self) -> None:
    """Called on any transition to flat (stop-driven or signal-driven)."""
    self.phase        = TSPhase.DISARMED
    self.peak_close   = 0.0
    self.trough_close = float("inf")

def update(self, bar: pd.Series) -> None:
    """Per-bar peak/trough ratchet on CLOSE only. Intrabar wicks ignored."""
    if self.phase == TSPhase.DISARMED:
        return
    c = float(bar["close"])
    if c > self.peak_close:
        self.peak_close = c
    if c < self.trough_close:
        self.trough_close = c

def stop_price(self, sign: int, bar_idx: int) -> Optional[float]:
    if self.phase == TSPhase.DISARMED or sign == 0:
        return None
    if self.phase == TSPhase.HARD:
        if pd.isna(self.atr_at_entry):
            return None
        offset = self.hard_stop_atr_mult * self.atr_at_entry
        # Fixed at entry; does NOT trail.
        return self.entry_price - offset if sign > 0 else self.entry_price + offset
    # RUNNER
    atr_now = float(self.atr_series.iloc[bar_idx])
    if pd.isna(atr_now):
        return None
    offset = self.runner_atr_mult * atr_now
    raw = (self.peak_close - offset) if sign > 0 else (self.trough_close + offset)
    if self.breakeven_floor:
        return max(raw, self.entry_price) if sign > 0 else min(raw, self.entry_price)
    return raw
```

### 2.5 Simulator-side detection of tranche 1 fill

The simulator's per-bar loop calls `ts.promote_to_runner()` on any bar where:

```python
prev_qty * new_qty > 0  AND  abs(new_qty) < abs(prev_qty)
```

i.e., same sign, smaller magnitude. This generalizes beyond the canonical 50%-50% case: any partial close from HARD phase promotes to RUNNER. The PRD's tranche structure is the canonical instance but not the only one.

### 2.6 Re-adding to a partially-closed position (out-of-scope edge case)

If a strategy emits a target whose magnitude *increases* while the position is in RUNNER phase (e.g., `0.5 → 1.0`), the simulator scales up the position but does NOT reset the phase. `peak_close` and `entry_price` are not recomputed. The PRD does not address this case; `mean_reversion_atr` does not do this. v0.4.0 documents the behavior as "undefined for new strategies; do not rely on it" in `docs/strategy_contract.md`.

### 2.7 Edge case: stop fills mid-bar via a gap

If a STOP order fills mid-bar at a price below the entry-bar's close, `entry_price` snapshots the *fill* price, not the close. The HARD-phase stop is `entry_price − 1.75 × atr_at_entry`. No special-casing — the strategy contracted for "stop at 1.75 ATR below entry price", and that's whatever entry price turned out to be.

---

## 3. MultiSymbolPortfolioSimulator

### 3.1 Per-bar loop

For each bar `i` over the shared time index, the simulator performs steps in this order. `symbols` is the set of universe symbols. Steps 1–6 run per-symbol; steps 7–10 run once per bar across the portfolio.

1. **Execute pending stop orders.** For each symbol with a non-None `pending_stop[sym]`, submit to that symbol's broker. Fills tagged `reason="trailing_stop"`. Position quantities updated. `stop_filled[sym] = True`. `ts[sym].disarm()`.
2. **Execute pending signal orders.** For each symbol with a non-None `pending_signal[sym]`, submit. If `stop_filled[sym]` is True, discard. Fills tagged `reason="signal"`. Position quantities updated.
3. **Compute `new_qty[sym]`** for each symbol.
4. **Tranche-state transitions.** For each symbol, compare `prev_qty[sym]` and `new_qty[sym]` (per §2.5). Call `ts[sym].reset()`, `.promote_to_runner()`, or `.disarm()` as appropriate.
5. **Update peak/trough.** For each symbol with non-DISARMED ts, call `ts[sym].update(bar)`.
6. **Update phase metadata.** `ctx.position_phase[sym]` and `ctx.bars_in_phase[sym]` reflect end-of-bar-`i` state. `ctx.recent_pnl` extended by bar-`i`'s portfolio PnL.
7. **Evaluate regime gates.** `RegimePolicy.update(bar_i, aux_data, ctx.recent_pnl)`. Sets `ctx.regime.book_flat` and per-gate state. If a gate trips this bar, all open positions are flattened (target = 0 forced for all symbols on the next bar's schedule).
8. **Strategy callback (per-bar mode).** For each symbol, call `strategy.signal_for_bar(symbol, i, data_panel, indicators_panel, ctx, params)`. Returns the target ∈ `[-1.0, 1.0]`. Strategies with `uses_per_bar = False` skip this step and read pre-computed signals at step 9.
9. **Apply portfolio-level scaling.** For each symbol's raw target, apply in order: regime gate (force 0 if `book_flat`), position cap, sector cap, risk budget. The result is `scaled_target[sym]`.
10. **Schedule orders for bar `i+1`.** For each symbol:
    - **Stop order**: if `ts[sym]` is armed and `new_qty[sym] ≠ 0`, compute `stop_price(sign, i+1)`. If not None, build a STOP order. Store in `pending_stop[sym]`.
    - **Signal order**: compute `delta = scaled_target[sym] * target_shares[sym] − new_qty[sym]`. If non-zero, build a MARKET (or LIMIT, if strategy emits a price column) order. Store in `pending_signal[sym]`.
11. **Mark-to-market** at bar.close for the portfolio.

### 3.2 Shared cash + equity

Single cash account and equity series across all symbols. Per-symbol `Position` objects track shares and average-cost; cash is debited/credited on each fill. `portfolio_equity[i] = cash[i] + Σ_sym (shares[sym] * close[sym, i])`.

### 3.3 RiskBudgetEnforcer

For each candidate new entry on bar `i+1`:

```
position_dollars = scaled_target[sym] * portfolio_equity[i] * position_cap_pct
stop_distance    = abs(close[sym, i] - ts[sym].stop_price(sign, i+1))
                   if ts armed else position_cap_pct * close[sym, i]  # conservative fallback
proposed_risk    = position_dollars / close[sym, i] * stop_distance

current_risk_used = Σ_open_sym (shares[s] * abs(close[s, i] - ts[s].stop_price(sign[s], i+1)))
                   for each open position with armed ts

if (current_risk_used + proposed_risk) / portfolio_equity[i] > risk_budget_pct:
    scaled_target[sym] = 0   # drop the entry
```

The risk budget releases incrementally as positions close (or partially close — when tranche 1 fills, its share of stop-distance is freed). The PRD's "no cooldown" requirement is satisfied: a new entry can use the freed budget on the same bar.

### 3.4 SectorCapEnforcer

For each candidate new entry:

```
deployed_per_sector[sector] = Σ_sym_in_sector (abs(shares[sym]) * close[sym, i])
deployed_total              = Σ_sym (abs(shares[sym]) * close[sym, i])
sector_pct                  = deployed_per_sector[sector] / max(deployed_total, 1e-9)

if sector_pct + proposed_position_pct > sector_cap_pct:
    scaled_target[sym] = 0   # drop or scale down
```

Sector membership is read from the universe.yaml's `sector` field. Missing sectors raise `ConfigError` at run start.

### 3.5 Volatility-targeted sizing

For each new entry, the strategy emits `target = 1.0` (full-size intent). The simulator computes the dollar position from:

```
realized_vol_20d  = close[sym].pct_change().rolling(20).std() * sqrt(252)
target_position_pct = portfolio.vol_target / realized_vol_20d[i]
target_position_pct = clip(target_position_pct, 0.0, portfolio.position_cap_pct)
dollars              = portfolio_equity[i] * target_position_pct
```

`vol_target` defaults to 0.12 (12% annualized). `position_cap_pct` defaults to 0.10 (10% per name). When `realized_vol_20d` is NaN during warmup, the simulator defers entry to the next bar. (No fallback to a default vol — explicit warmup, predictable behavior.)

The 70%-deployed/30%-cash split is enforced by capping total deployed dollars at `portfolio_equity[i] * (1 − cash_reserve_pct)`. New entries that would exceed this cap are scaled down or dropped.

---

## 4. RegimePolicy

A new class `RegimePolicy` in `backtester/engine/regime.py` owns the three gates. Composed of three independently-toggleable sub-gates. `book_flat = gate.spy_tripped OR gate.vix_tripped OR gate.circuit_tripped`. Strategy reads `ctx.regime` for diagnostics only — cannot bypass.

### 4.1 SPY 200-EMA gate

```python
spy_ema      = aux_data["SPY"]["close"].ewm(span=200, adjust=False).mean()
spy_close    = aux_data["SPY"]["close"]
trip_value   = spy_ema[i] * (1 + regimes.spy_ema.trip_pct)     # negative pct → trip below
resume_value = spy_ema[i] * (1 + regimes.spy_ema.resume_pct)   # positive pct → resume above

if not tripped: trip   if spy_close[i] < trip_value
if tripped:     resume if spy_close[i] > resume_value
```

Defaults: `trip_pct = -0.02`, `resume_pct = 0.02`. Hysteresis prevents flapping.

### 4.2 VIX hysteresis gate

```python
vix_close = aux_data["^VIX"]["close"]

if not tripped: trip   if vix_close.tail(2) all > regimes.vix.trip_threshold      # 2 consec > 30
if tripped:     resume if vix_close.tail(3) all < regimes.vix.resume_threshold    # 3 consec < 25
```

Defaults: `trip_threshold = 30`, `trip_consec = 2`, `resume_threshold = 25`, `resume_consec = 3`. Both `consec` counts are configurable. Validation enforces `resume_threshold < trip_threshold`.

### 4.3 Strategy circuit breaker

```python
rolling_pnl_pct = ctx.recent_pnl.rolling(regimes.circuit_breaker.pnl_window_days).sum() / initial_cash

if not tripped: trip if rolling_pnl_pct[i] <= regimes.circuit_breaker.trip_pct
if tripped:
    if bars_since_trip >= regimes.circuit_breaker.pause_days:
        resume    # full size on day 11 — PRD literal, no phased re-entry in v0.4.0
```

Defaults: `pnl_window_days = 20`, `trip_pct = -0.05`, `pause_days = 10`. The "full size on day 11" semantics are the PRD literal; phased re-entry is deferred to v0.4.1 if WFO shows ratchet-down clusters at regime onset. Documented as such.

### 4.4 Flatten-on-trip

When any gate transitions from not-tripped to tripped on bar `i`:
- All open positions get `scaled_target[sym] = 0` for the bar-`i+1` schedule.
- `pending_stop` orders are left active — they fire normally if breached intraday on bar `i+1`. Otherwise the bar-`i+1` market-on-open exit closes the position.
- New entry signals are suppressed for as long as the gate is tripped.

When all gates resume:
- New entries resume normally on the next bar.
- No bulk re-entry — the strategy emits entries per its own signal logic, one symbol at a time as opportunities appear.

---

## 5. Data layer

### 5.1 yfinance loader

A new `source: yfinance` option on `DataConfig`. Wired into `backtester/data/loader.py`:

```python
def load_symbol(symbol, source, root, start, end, *, auto_adjust=True, require_volume=True):
    if source == "csv":
        return _load_csv(symbol, root, start, end)                               # unchanged
    elif source == "yfinance":
        return _load_yfinance_cached(symbol, root, start, end,
                                     auto_adjust=auto_adjust,
                                     require_volume=require_volume)
    else:
        raise ConfigError(f"unknown data.source: {source}")
```

**Cache-on-miss semantics.** `_load_yfinance_cached`:

1. Check `{root}/{symbol}.csv`. If present, read CSV. Validate it covers `[start, end]`. If yes, slice and return. If no, raise `DataError` with a "rm the file to re-fetch" message.
2. If absent, call `yfinance.download(symbol, period="max", auto_adjust=auto_adjust, progress=False)`. Write the full history to `{root}/{symbol}.csv`. Slice to `[start, end]` and return.

No silent re-fetch. Cache invalidation is explicit (user deletes the CSV).

**Adjustment contract.** When `source: yfinance` with `auto_adjust=True` (the framework default), ALL of `open`, `high`, `low`, and `close` columns are adjusted for splits and dividends. Volume is unadjusted (yfinance behavior). This is the framework's adjustment contract — downstream code (ATR, stop logic, range metrics, screening) reads adjusted OHLC uniformly and does not need to know the provider. To opt out, set `data.auto_adjust: false`; the loader then stores raw OHLC and the caller handles corporate actions.

**`^VIX` quirk.** Index-style symbols have no volume. `_load_yfinance_cached` accepts `require_volume=False` and fills the volume column with zeros. `validate_ohlcv` gets a matching `strict_volume: bool = True` arg to allow the relaxed path. The runner passes `require_volume=False` for symbols declared in `data.aux_symbols`.

**Dependency.** `yfinance` enters `pyproject.toml` as an optional extras group:

```toml
[project.optional-dependencies]
data = ["yfinance>=0.2.40"]
```

Tests that exercise yfinance code paths are marked `@pytest.mark.requires_yfinance` and skipped if the import fails. The core package stays network-free.

**CI determinism.** After the first developer fetches the 17 tickers (15 universe + SPY + ^VIX) covering 2015-2025, those CSVs are committed to git as test fixtures (~250 KB × 17 ≈ 4 MB). CI runs against committed fixtures; yfinance never executes in CI. The `_load_yfinance_cached` code path is exercised by a single unit test that monkeypatches `yfinance.download`.

### 5.2 Synthetic-data generator relocation

`scripts/generate_sample_data.py` is not retired — it remains useful for unit-test fixtures requiring deterministic synthetic OHLCV. Output moves from `data/raw/` to `data/synth/`. `tests/integration/test_backwards_compat.py` is updated to point at `data/synth/SPY.csv`; its v0.3.0 golden numbers are recaptured against the relocated (but byte-identical) synthetic SPY. Every other test points at `data/raw/` with real data.

### 5.3 Sector map

A committed lookup table `data/sector_map.csv` (~17 rows for the PRD universe, expandable) is the canonical source of sector membership.

```csv
symbol,sector
TSLA,Auto
NVDA,Semis
AMD,Semis
COIN,Crypto
GOOGL,BigTech
MSTR,Crypto
XPEV,Auto
NIO,Auto
PLTR,Software
SMCI,Semis
SHOP,Software
W,Consumer
META,BigTech
NFLX,Media
SPY,Index
^VIX,Index
```

`scripts/screen_universe.py` looks each candidate up in this file and writes the result inline into `universe_candidates.yaml`. Tickers absent from `sector_map.csv` receive `sector: unknown` and the CLI emits a warning. Hand-edits in `universe.yaml` override `sector_map.csv` if they conflict (universe.yaml is downstream). The CSV is committed because (a) it's stable enough, (b) the alternative — fetching from `yfinance.Ticker(symbol).info` — is non-deterministic and adds a network call.

---

## 6. Universe screening CLI (`scripts/screen_universe.py`)

### 6.1 Purpose

Resolves PRD item 1 (range-width measurement) and item 2 (trend filter). Pre-screens a candidate ticker list and emits `configs/universe_candidates.yaml` ranked by mean-reversion fitness.

### 6.2 Invocation

```
$ python scripts/screen_universe.py \
    --candidates configs/universe_candidates_seed.txt \
    --start 2023-01-01 --end 2025-12-31 \
    --out configs/universe_candidates.yaml \
    --top 20
```

`--candidates` is a flat newline-delimited list of tickers. The CLI calls `load_symbol(source="yfinance", ...)` for each, computing metrics over the screening window.

### 6.3 Metrics

| Metric | Definition |
|---|---|
| `range_p10_p90_63d` | rolling 63-bar `(close.p90 − close.p10)`; reported as median across the screening window |
| `atr_tr_20` | TR-based ATR with period 20 from `backtester/engine/atr.py`; reported as median |
| `range_atr_ratio` | `range_p10_p90_63d / atr_tr_20`; the comparable mean-reversion fitness metric. Higher = wider range relative to typical bar movement |
| `slope_200d_pct_per_day` | OLS slope of `ln(close)` on bar index over a rolling 200-bar window, transformed to daily-percent via `expm1(slope_log) = e^slope − 1`; reported as median. For small slopes (`|slope| < 0.01`) this is numerically indistinguishable from the raw log-slope; for larger slopes it's the correct compounded-percent figure. The trend filter's threshold (0.002 ≈ 0.2%/day) is applied to this percent value. |
| `r_squared_200d` | OLS R² over the same 200-bar window; reported as median |

### 6.4 Filters

Applied in order:

1. **Min data length.** ≥ 504 bars (~2 years). Drop tickers with less.
2. **Trend filter (PRD item 2).** Drop tickers where `abs(slope_200d_pct_per_day) > 0.002` AND `r_squared_200d > 0.4`. Both conditions must hold — a noisy ticker with a high slope but low R² survives.
3. **Min range/ATR ratio.** Drop tickers with `range_atr_ratio < 5.0`. Below this, the range is too tight relative to noise.

### 6.5 Output

```yaml
# generated by scripts/screen_universe.py on 2026-05-14
# screening window: 2023-01-01 to 2025-12-31
# filters: |slope| < 0.2%/d AND R² < 0.4; range/atr ≥ 5.0
universe:
  TSLA:  {sector: Auto,     range_atr_ratio: 9.2, slope_200d: 0.05, r_squared_200d: 0.18}
  NVDA:  {sector: Semis,    range_atr_ratio: 7.8, slope_200d: 0.18, r_squared_200d: 0.31}
  ...
```

Diagnostic fields are advisory; only `sector` and any `overrides` block in the hand-curated `universe.yaml` are consumed by the runner.

### 6.6 Runtime trend gate (layer 2)

The CLI is layer 1 (pre-selection). Layer 2 is the strategy's per-bar `runtime_trend_threshold` (§1.1). The CLI keeps a name out of the universe; the runtime gate keeps an in-universe name from taking a new entry if it develops a trend mid-run. Existing positions are NOT closed by the runtime gate — only new entries are suppressed.

---

## 7. Configuration

### 7.1 Run YAML — `configs/backtests/mean_rev_v04.yaml`

```yaml
run_name: mean_rev_v04
strategy: mean_reversion_atr

universe_path: configs/universe.yaml      # resolved relative to this run YAML

data:
  source: yfinance
  root: data/raw
  start: '2015-01-02'
  end: '2025-12-31'
  timeframe: 1d
  auto_adjust: true
  aux_symbols: [SPY, '^VIX']
  # Universe symbols come from universe.yaml; data.symbols is ignored for multi-symbol runs.

strategy_params:
  entry_atr_mult:           1.25
  mean_lookback:            10
  atr_lookback:             20
  time_stop_days:           7
  runner_time_stop_days:    12
  runner_ceiling_atr_mult:  1.25
  runtime_trend_threshold:  0.0025

execution:
  initial_cash:             100000
  commission_bps:           2
  slippage_bps:             5
  allow_fractional:         false
  allow_short:              false
  hard_stop_atr_mult:       1.75
  runner_atr_mult:          2.5
  breakeven_floor:          true
  tranche_stop_atr_period:  20

portfolio:
  sizing_mode:              vol_targeted    # new mode in v0.4.0
  vol_target:               0.12
  position_cap_pct:         0.10
  cash_reserve_pct:         0.30
  risk_budget_pct:          0.06
  sector_cap_pct:           0.50

regimes:
  spy_ema:
    enabled:                true
    ema_lookback:           200
    trip_pct:               -0.02
    resume_pct:              0.02
  vix:
    enabled:                true
    trip_threshold:         30
    trip_consec:             2
    resume_threshold:       25
    resume_consec:           3
  circuit_breaker:
    enabled:                true
    pnl_window_days:        20
    trip_pct:               -0.05
    pause_days:             10

output_root: output/runs
```

### 7.2 Universe YAML — `configs/universe.yaml`

```yaml
universe:
  TSLA:  {sector: Auto,     overrides: {entry_atr_mult: 1.5}}
  NVDA:  {sector: Semis,    overrides: {}}
  AMD:   {sector: Semis,    overrides: {}}
  COIN:  {sector: Crypto,   overrides: {entry_atr_mult: 1.5}}
  GOOGL: {sector: BigTech,  overrides: {}}
  MSTR:  {sector: Crypto,   overrides: {entry_atr_mult: 1.75}}
  XPEV:  {sector: Auto,     overrides: {}}
  NIO:   {sector: Auto,     overrides: {}}
  PLTR:  {sector: Software, overrides: {mean_lookback: 14}}
  SMCI:  {sector: Semis,    overrides: {entry_atr_mult: 1.5}}
  SHOP:  {sector: Software, overrides: {}}
  W:     {sector: Consumer, overrides: {}}
  META:  {sector: BigTech,  overrides: {}}
  NFLX:  {sector: Media,    overrides: {}}
  # 15th name to be selected after running scripts/screen_universe.py
```

**Loader behavior.** `load_universe_config(path)`:
1. Reads the YAML.
2. For each symbol, looks up its sector in `data/sector_map.csv`. If `universe.yaml` declares a `sector`, that overrides the CSV.
3. For each symbol, materializes its effective `strategy_params` as `global_params ∪ overrides` (per-name overrides win).
4. Returns `dict[symbol, ResolvedSymbolConfig]` consumed by `MultiSymbolPortfolioSimulator`.

`config_resolved.yaml` written by `ArtifactWriter` materializes a flat, fully-expanded view: every per-name override resolved against global defaults, every regime gate's effective values explicit. Reproducible from this single artifact.

### 7.3 New `DataConfig` fields

```python
@dataclass(slots=True)
class DataConfig:
    source:        str                                  # "csv" or "yfinance" (new)
    root:          str
    start:         str
    end:           str
    timeframe:     str
    symbols:       list[str]      = field(default_factory=list)
    auto_adjust:   bool           = True                # new in v0.4.0
    aux_symbols:   list[str]      = field(default_factory=list)   # new in v0.4.0
```

### 7.4 New `ExecutionConfig` fields

```python
@dataclass(slots=True)
class ExecutionConfig:
    # existing v0.3.0 fields ...
    hard_stop_atr_mult:        Optional[float] = None    # new
    runner_atr_mult:           Optional[float] = None    # new
    breakeven_floor:           bool            = True    # new
    tranche_stop_atr_period:   int             = 20      # new
```

### 7.5 New `PortfolioConfig` fields

```python
@dataclass(slots=True)
class PortfolioConfig:
    # existing v0.3.0 fields ...
    sizing_mode:        str   = "percent_equity"    # adds "vol_targeted"
    vol_target:         float = 0.12                # new; ignored if sizing_mode != "vol_targeted"
    position_cap_pct:   float = 1.0                 # new
    cash_reserve_pct:   float = 0.0                 # new
    risk_budget_pct:    float = 1.0                 # new
    sector_cap_pct:     float = 1.0                 # new
```

### 7.6 New top-level `RunConfig` fields

```python
@dataclass(slots=True)
class RunConfig:
    # existing v0.3.0 fields ...
    universe_path:  Optional[Path]               = None      # new
    regimes:        Optional[RegimesConfig]      = None      # new
```

### 7.7 New `RegimesConfig`

```python
@dataclass(slots=True)
class SpyEmaRegimeConfig:
    enabled:      bool  = False
    ema_lookback: int   = 200
    trip_pct:     float = -0.02
    resume_pct:   float = 0.02

@dataclass(slots=True)
class VixRegimeConfig:
    enabled:           bool  = False
    trip_threshold:    float = 30.0
    trip_consec:       int   = 2
    resume_threshold:  float = 25.0
    resume_consec:     int   = 3

@dataclass(slots=True)
class CircuitBreakerConfig:
    enabled:          bool  = False
    pnl_window_days:  int   = 20
    trip_pct:         float = -0.05
    pause_days:       int   = 10

@dataclass(slots=True)
class RegimesConfig:
    spy_ema:         SpyEmaRegimeConfig    = field(default_factory=SpyEmaRegimeConfig)
    vix:             VixRegimeConfig       = field(default_factory=VixRegimeConfig)
    circuit_breaker: CircuitBreakerConfig  = field(default_factory=CircuitBreakerConfig)
```

### 7.8 Validation rules added

All raise `ConfigError`. Numbered for cross-reference with test names.

1. `hard_stop_atr_mult` and `runner_atr_mult` are both-or-neither. One set without the other raises.
2. `trailing_stop_pct` or `trailing_stop_atr_mult` set AND `hard_stop_atr_mult` set → mutually-exclusive error.
3. `hard_stop_atr_mult > 0`, `runner_atr_mult > 0`.
4. `tranche_stop_atr_period >= 2`.
5. `position_cap_pct ∈ (0, 1]`.
6. `cash_reserve_pct ∈ [0, 1)`.
7. `risk_budget_pct ∈ (0, 1]`.
8. `sector_cap_pct ∈ (0, 1]`.
9. `vol_target > 0` (only checked when `sizing_mode == "vol_targeted"`).
10. `regimes.circuit_breaker.pause_days >= 0`.
11. `regimes.vix.resume_threshold < regimes.vix.trip_threshold`.
12. `regimes.spy_ema.trip_pct <= 0` and `regimes.spy_ema.resume_pct >= 0`.
13. `regimes.vix.trip_consec >= 1` and `regimes.vix.resume_consec >= 1`.
14. `regimes.circuit_breaker.trip_pct < 0`.
15. `universe_path` exists and parses. Each ticker's `overrides` keys ⊆ `strategy_params` keys.
16. `data.aux_symbols` is non-empty when any `regimes.*.enabled` is True (SPY required if `spy_ema.enabled`, ^VIX required if `vix.enabled`).
17. Every symbol in `universe.yaml` resolves to a sector via `sector_map.csv` or inline.
18. When `universe_path` is set, `data.symbols` MUST be empty. Setting both raises `ConfigError("data.symbols and universe_path are mutually exclusive; universe.yaml is the single source of symbol membership for multi-symbol runs")`.

---

## 8. File layout

| File | Action |
|---|---|
| **Strategy** | |
| `strategies/mean_reversion_atr.py` | create |
| `backtester/strategies/registry.py` | modify — register `mean_reversion_atr` |
| `backtester/strategies/base.py` | modify — add `uses_multi_symbol`, `uses_per_bar` class attributes |
| **Engine** | |
| `backtester/engine/tranche_stop.py` | create (TSPhase, TrancheStopState) |
| `backtester/engine/regime.py` | create (RegimePolicy, SpyEmaGate, VixGate, CircuitBreakerGate) |
| `backtester/engine/risk_budget.py` | create (RiskBudgetEnforcer) |
| `backtester/engine/sector_cap.py` | create (SectorCapEnforcer) |
| `backtester/engine/multi_portfolio.py` | create (MultiSymbolPortfolioSimulator) |
| `backtester/engine/multi_backtest_engine.py` | create (MultiSymbolBacktestEngine) |
| `backtester/core/types.py` | modify — extend StrategyContext with `position_phase`, `bars_in_phase`, `recent_pnl`, `regime` |
| `backtester/engine/trailing_stop.py` | unchanged |
| `backtester/engine/portfolio.py` | unchanged |
| **Data** | |
| `backtester/data/yfinance_loader.py` | create |
| `backtester/data/loader.py` | modify — route by `source` |
| `backtester/data/validators.py` | modify — `strict_volume` flag |
| `data/sector_map.csv` | create |
| `data/raw/SPY.csv`, `data/raw/AAPL.csv`, `data/raw/<universe>.csv`, `data/raw/^VIX.csv` | replace (synthetic → real, committed as fixtures) |
| `data/synth/SPY.csv`, `data/synth/AAPL.csv` | create (relocated synthetic) |
| `scripts/generate_sample_data.py` | modify — write to `data/synth/` |
| **Config** | |
| `backtester/config/models.py` | modify — new DataConfig, ExecutionConfig, PortfolioConfig, RegimesConfig fields; new RunConfig fields |
| `backtester/config/loader.py` | modify — load universe.yaml when `universe_path` set |
| `backtester/config/universe.py` | create (universe + sector_map loader) |
| `backtester/config/validation.py` | modify — 17 new rules |
| `configs/backtests/mean_rev_v04.yaml` | create |
| `configs/universe.yaml` | create |
| `configs/optimize/mean_rev_v04_grid.yaml` | create |
| `configs/wfo/mean_rev_v04_wfo.yaml` | create |
| `configs/universe_candidates_seed.txt` | create (seed list for screen_universe.py) |
| **Optimizer** | |
| `backtester/optimize/grid_search.py` | modify — add `sampling: lhs` mode + `random_seed` + `random_n` |
| `backtester/optimize/lhs_sampler.py` | create (discrete-LHS over index positions) |
| **Runners** | |
| `backtester/runners/run_batch.py` | modify — route through MultiSymbolBacktestEngine when `strategy.uses_multi_symbol` |
| `backtester/runners/run_wfo.py` | modify — same routing |
| `backtester/runners/run_backtest.py` | unchanged (single-symbol; falls back to v0.3.0 path) |
| **CLI** | |
| `scripts/screen_universe.py` | create |
| **Artifacts** | |
| `backtester/io/artifacts.py` | modify — write `portfolio_equity_curve.csv`, `batch_summary.json`, per-symbol bundles, materialized `config_resolved.yaml` |
| **Tests** | |
| `tests/unit/test_tranche_stop.py` | create — 14 tests |
| `tests/unit/test_multi_symbol_simulator.py` | create — 18 tests |
| `tests/unit/test_regime_policy.py` | create — 10 tests |
| `tests/unit/test_yfinance_loader.py` | create — 5 tests |
| `tests/unit/test_screen_universe.py` | create — 6 tests |
| `tests/unit/test_strategy_mean_reversion_atr.py` | create — 14 tests |
| `tests/unit/test_universe_loader.py` | create — 4 tests |
| `tests/unit/test_optimizer_lhs.py` | create — 3 tests |
| `tests/unit/test_sector_map.py` | create — 2 tests |
| `tests/unit/test_config_validation.py` | append — 18 new tests (one per rule in §7.8) |
| `tests/integration/test_stress_windows.py` | create — 4 parametrized tests |
| `tests/integration/test_held_out_2022_2025.py` | create — 1 test |
| `tests/integration/test_screen_universe_cli.py` | create — 1 test |
| `tests/integration/test_run_batch_cli.py` | append — 1 test (multi-symbol mean-reversion smoke) |
| `tests/integration/test_run_wfo_cli.py` | append — 1 test (mean-reversion WFO smoke) |
| `tests/integration/test_backwards_compat.py` | modify — point at `data/synth/SPY.csv`; recapture golden |
| **Docs** | |
| `docs/strategy_contract.md` | modify — float signal, aux_data, per-bar callback, StrategyContext extensions |
| `docs/runbook.md` | modify — v0.4.0 limitations, performance-gate flip workflow |
| `README.md` | modify — Execution model + multi-symbol + regime gates |
| **Project** | |
| `pyproject.toml` | bump 0.3.0 → 0.4.0; add `[project.optional-dependencies.data] = ["yfinance>=0.2.40"]` |

---

## 9. Tests

Baseline after v0.3.0: 202 tests. v0.4.0 adds ~87, target ~289.

### 9.1 Unit — `tests/unit/test_tranche_stop.py` (14 tests)

1. `test_disarmed_by_default`
2. `test_reset_sets_hard_phase_and_snapshots_entry`
3. `test_hard_stop_uses_atr_at_entry_not_current_atr`
4. `test_hard_stop_does_not_trail`
5. `test_promote_to_runner_keeps_peak_close`
6. `test_promote_to_runner_idempotent`
7. `test_runner_trail_uses_peak_close_not_peak_high`
8. `test_runner_trail_clipped_by_breakeven_floor`
9. `test_runner_trail_no_floor_when_disabled`
10. `test_disarm_clears_peak_close_and_phase`
11. `test_update_only_reads_close_field`
12. `test_stop_price_returns_none_when_atr_nan`
13. `test_short_symmetric_trough_close_and_breakeven_ceiling`
14. `test_re_adding_during_runner_does_not_reset_phase`

### 9.2 Unit — `tests/unit/test_multi_symbol_simulator.py` (18 tests)

1. `test_shared_cash_debited_on_buy_credited_on_sell`
2. `test_portfolio_equity_sums_cash_plus_positions`
3. `test_two_symbol_independent_entries_dont_interfere`
4. `test_simultaneous_entries_compete_for_risk_budget`
5. `test_position_cap_clips_oversized_signal`
6. `test_cash_reserve_cap_drops_late_entries`
7. `test_sector_cap_blocks_third_entry_in_full_sector`
8. `test_risk_budget_released_on_partial_close`
9. `test_risk_budget_released_on_full_exit`
10. `test_vol_targeted_sizing_uses_realized_vol_20d`
11. `test_vol_targeted_sizing_clips_at_position_cap`
12. `test_vol_targeted_sizing_defers_entry_during_warmup`
13. `test_promote_to_runner_called_on_partial_close`
14. `test_disarm_called_on_full_exit`
15. `test_pending_stop_per_symbol_independent`
16. `test_stop_wins_over_signal_same_bar_per_symbol`
17. `test_position_phase_finalized_before_strategy_callback`
18. `test_per_bar_callback_skipped_for_legacy_strategies`

### 9.3 Unit — `tests/unit/test_regime_policy.py` (10 tests)

1. `test_spy_ema_trips_below_threshold`
2. `test_spy_ema_resumes_above_threshold_hysteresis`
3. `test_vix_requires_two_consecutive_above_30`
4. `test_vix_resume_requires_three_consecutive_below_25`
5. `test_circuit_breaker_trips_on_minus_5_pct_rolling_20d`
6. `test_circuit_breaker_resumes_after_pause_days`
7. `test_circuit_breaker_resumes_at_full_size_not_phased`
8. `test_book_flat_is_disjunction_across_gates`
9. `test_disabled_gate_never_trips`
10. `test_flatten_on_trip_emits_zero_target_for_all_open`

### 9.4 Unit — `tests/unit/test_yfinance_loader.py` (5 tests)

1. `test_loads_from_cache_when_csv_present` (no network)
2. `test_fetches_via_yfinance_when_csv_absent` (monkeypatched)
3. `test_raises_data_error_when_cache_does_not_cover_range`
4. `test_auto_adjust_true_returns_adjusted_ohlc`
5. `test_require_volume_false_allows_zero_volume_for_vix`

### 9.5 Unit — `tests/unit/test_screen_universe.py` (6 tests)

1. `test_range_atr_ratio_definition` — synthetic 100-bar series, hand-computed
2. `test_slope_200d_pct_per_day_uses_expm1` — slope of `ln(close)` transformed to percent
3. `test_trend_filter_requires_both_slope_and_r_squared`
4. `test_min_data_length_filter`
5. `test_emits_unknown_sector_with_warning`
6. `test_top_n_caps_output`

### 9.6 Unit — `tests/unit/test_strategy_mean_reversion_atr.py` (14 tests)

1. `test_entry_fires_at_125_atr_below_mean10`
2. `test_entry_suppressed_when_phase_not_disarmed`
3. `test_entry_suppressed_when_trend_active`
4. `test_entry_suppressed_when_regime_book_flat`
5. `test_tranche_1_emits_half_target_at_mean_touch`
6. `test_hard_phase_time_stop_at_7_days`
7. `test_runner_phase_time_stop_at_12_days`
8. `test_runner_hard_ceiling_at_mean_plus_125_atr`
9. `test_strategy_does_not_close_position_on_runtime_trend_gate`
10. `test_signal_shifted_one_bar`
11. `test_warmup_bars_correct`
12. `test_emits_no_signal_during_warmup`
13. `test_per_bar_callback_signature`
14. `test_position_phase_visible_in_ctx`

### 9.7 Unit — `tests/unit/test_universe_loader.py` (4 tests)

1. `test_loads_universe_yaml_with_per_name_overrides`
2. `test_overrides_keys_must_be_subset_of_strategy_params`
3. `test_inline_sector_overrides_sector_map_csv`
4. `test_missing_sector_raises_config_error`

### 9.8 Unit — `tests/unit/test_optimizer_lhs.py` (3 tests)

1. `test_lhs_index_position_sampling_balanced` — for a 4-element list and random_n=8, each index appears 2× (±1 due to flooring)
2. `test_lhs_seed_determinism`
3. `test_lhs_rejects_random_n_larger_than_cartesian_product`

### 9.9 Unit — `tests/unit/test_sector_map.py` (2 tests)

1. `test_sector_map_csv_parses`
2. `test_every_universe_symbol_has_sector_entry`

### 9.10 Unit — `tests/unit/test_config_validation.py` (18 new tests)

One per rule in §7.8. Naming convention: `test_validation_rule_NN_<description>`.

### 9.11 Integration — `tests/integration/test_stress_windows.py` (4 parametrized tests)

```python
STRESS_WINDOWS = [
    ("2020-covid",      "2020-02-15", "2020-04-30"),
    ("2022-bear-cycle", "2021-11-01", "2022-10-31"),
    ("2024-aug-unwind", "2024-07-15", "2024-09-15"),
    ("2025-apr",        "2025-03-15", "2025-05-15"),
]

@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
@pytest.mark.parametrize("label,start,end", STRESS_WINDOWS)
def test_stress_window_drawdown(tmp_path, label, start, end):
    cfg = _load_mean_rev_config(start=start, end=end, output=tmp_path)
    rc = run_batch_main(["--config", str(cfg)])
    assert rc == 0, "structural correctness gate"
    summary = json.loads((tmp_path / "batch_summary.json").read_text())
    _write_metrics(tmp_path / "metrics.json", summary)   # always reports
    assert summary["portfolio_max_drawdown"] > -0.09, (
        f"{label}: DD {summary['portfolio_max_drawdown']:.4f} exceeded -9%"
    )
```

Each window is independently flippable from xfail to hard-assert by removing the `xfail` marker.

### 9.12 Integration — `tests/integration/test_held_out_2022_2025.py` (1 test)

```python
@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
def test_held_out_2022_2025(tmp_path):
    cfg = _load_mean_rev_config(start="2022-01-01", end="2025-12-31", output=tmp_path)
    rc = run_batch_main(["--config", str(cfg)])
    assert rc == 0
    summary = json.loads((tmp_path / "batch_summary.json").read_text())
    _write_metrics(tmp_path / "metrics.json", summary)
    assert summary["portfolio_max_drawdown"] > -0.09, "held-out DD breach"
    assert summary["portfolio_total_return"] > 0.15, "held-out return under 15%"
```

### 9.13 Integration — CLI smokes (4 new/append tests)

1. `tests/integration/test_run_batch_cli.py::test_multi_symbol_mean_reversion_smoke` — run on a 3-symbol subset over 2024-2025; assert exit 0, batch_summary.json has all 3 symbols, portfolio_equity_curve.csv exists.
2. `tests/integration/test_run_wfo_cli.py::test_mean_reversion_wfo_smoke` — small grid (2×2×2 = 8 combos) over 900 bars; assert window_results.json exists.
3. `tests/integration/test_screen_universe_cli.py::test_screen_universe_smoke` — 5-ticker seed, assert exit 0 and parseable output.
4. `tests/integration/test_backwards_compat.py::test_sma_cross_synth_spy_unchanged` — modified to point at `data/synth/SPY.csv`; golden numbers recaptured.

### 9.14 Performance-gate flip workflow

Documented in `docs/runbook.md`. The v0.4.0 framework ships with all strategy-performance assertions wrapped in `@pytest.mark.xfail(strict=False)`. The tests always:

- Run the backtest end-to-end (structural correctness).
- Parse metrics from `summary.json` / `batch_summary.json`.
- Write metrics to `output/runs/{test_name}/metrics.json` for inspection.

To convert a target to a hard gate, the user removes the `xfail` marker in source. No test rewrite. CI stays green during tuning; performance regressions surface as assertion failures only after the user flips the marker. This separates framework correctness from strategy tuning.

### 9.15 Total

88 new tests. Target: 290 passing.

---

## 10. Acceptance criteria

### 10.1 Framework-correctness gates (block release)

1. `pytest -q` passes — ~290 tests, zero regressions in the 202 v0.3.0 baseline. The 5 new xfail-marked performance tests are expected-xfail; they count as pass.
2. `python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml` against the relocated `data/synth/SPY.csv` produces output byte-identical to v0.3.0's recaptured baseline (the modified `test_backwards_compat.py` enforces this).
3. `python -m backtester.runners.run_batch --config configs/backtests/mean_rev_v04.yaml` exits 0 against the committed real-data fixtures. Produces `batch_summary.json` covering all 15 names and a `portfolio_equity_curve.csv` artifact (new for multi-symbol runs).
4. `python -m backtester.runners.run_wfo --config configs/wfo/mean_rev_v04_wfo.yaml` exits 0 with `sampling: lhs` and `random_n: 200` per window. `window_results.json` materializes the 6-dim param surface per window.
5. `python scripts/screen_universe.py --candidates configs/universe_candidates_seed.txt --top 20` exits 0 and writes a parseable `universe_candidates.yaml`.
6. `pyproject.toml` version is `0.4.0`. Git tag `v0.4.0` created. Push only after user confirms.
7. `README.md`, `docs/strategy_contract.md`, `docs/runbook.md` updated to reflect the contract changes.
8. v0.3.0 backwards compat — every existing config in `configs/backtests/`, `configs/optimize/`, `configs/wfo/` continues to run end-to-end with no warnings (verified by the existing test suite passing untouched).

### 10.2 Strategy-performance gates (reported, not asserted)

The PRD's targets — "in-sample Calmar > 2.5", "held-out 2022-2025 return > 15% AND no 9% DD breach", "no 9% DD breach in stress windows" — are NOT release blockers in v0.4.0. They are wired as xfail-by-default tests (§9.11–9.12) that always report metrics to `output/runs/{test}/metrics.json`. The framework ships when §10.1 passes. Whether the strategy hits the PRD's numbers is a tuning question, addressable by editing `mean_rev_v04.yaml`, `universe.yaml`, and the WFO grid — not by patching the framework.

When the strategy is tuned and the user is satisfied with the metrics, they remove the `xfail` markers in source to convert the assertions into hard gates. This is documented as the "performance gate flip" workflow in `docs/runbook.md`.

---

## 11. Out of scope

Deliberately deferred to v0.4.x or v0.5.0:

- **Phased circuit-breaker re-entry.** v0.4.0 uses the PRD literal: full size on day 11. If WFO surfaces ratchet-down clusters at regime onset, phased re-entry (50% on day 11, 100% on day 21) becomes a v0.4.1 follow-up.
- **Strategy-emitted stops.** Strategies cannot emit a `stop_column`. Stops remain execution-layer (`TrancheStopState` for new strategies, `TrailingStopState` for v0.3.0 strategies).
- **Time-based stops in execution layer.** Time stops live in the strategy (`time_stop_days`, `runner_time_stop_days`). The execution layer does not implement "exit after N bars without progress" as a generic option.
- **Asymmetric long/short hard/runner ratios.** `TrancheStopState` parameters apply symmetrically. Wider stops on shorts would duplicate fields.
- **Borrow cost / short-interest modeling.** Inherited from v0.2.0 limitations. mean_reversion_atr is long-only, so this doesn't bite v0.4.0 directly, but the framework still can't price it for any future short strategy.
- **Multi-timeframe data.** All symbols and aux symbols share `timeframe: 1d`. Mixed timeframes are out of scope.
- **Continuous-bound LHS sampling.** Optimizer's LHS mode samples discrete-list index positions only. Strategies with truly continuous param ranges would need a separate sampler.
- **Real-time / paper-trading mode.** Backtester remains offline.
- **Phased universe rebalancing.** The universe is fixed for the duration of a run. Adding/dropping names mid-run is a v0.5.0 concern.
- **`yfinance` retries, rate limiting, or alternative providers.** First-fetch is best-effort; failures surface to the user immediately. No retry/backoff machinery.
- **Sector membership changes over time.** `sector_map.csv` is a static snapshot. A ticker that changed sectors during 2015-2025 is mapped to its current sector for the whole window.
- **WFO over the regime / sizing / risk-budget surface.** v0.4.0's WFO grid is the 6-dim strategy surface (entry, mean, time-stops, runner mults, ceiling). Regime constants (200d, VIX 30/25, −5%) and sizing constants (vol_target, position_cap, risk_budget, sector_cap) stay fixed in v0.4.0.

---

## 12. Resolved PRD open items

| PRD item | Resolution |
|---|---|
| 1. Range-width measurement method | `scripts/screen_universe.py` with `range_p10_p90_63d / atr_tr_20` ratio. §6. |
| 2. Trend/range gate before vol screen | Both layers: CLI screening (`scripts/screen_universe.py`) + runtime per-bar trend gate in strategy (`runtime_trend_threshold`). §6.6. |
| 3. ATR uses true range | Reuses v0.3.0 `backtester/engine/atr.py`. Confirmed. |
| 4. Tranche 1 mean drift | Accepted as-is. Runner offsets at the position level. §1.4. |
| 5. Circuit-breaker re-entry | Full size on day 11 (PRD literal). Phased re-entry deferred to v0.4.1 if WFO needs it. §4.3, §11. |
