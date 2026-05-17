# ml_supertrend Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the signal core of the "Machine Learning Supertrend [Aslan]" TradingView Pine Script into a curated, vectorized daily strategy that runs in this backtester.

**Architecture:** One new strategy module (`strategies/ml_supertrend.py`) with four module-level pure helper functions (source resolution, smoothed true range, SuperTrend trend state, Wilder RSI) and a `BaseStrategy` subclass. `indicators()` builds all derived columns vectorized; `generate_signals()` runs a single sequential numpy loop that handles signal spacing, the reversal-mode flag latches, and the stop-and-reverse position. The adaptive "ML" engine from the Pine Script is intentionally not ported.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. Follows the existing curated-strategy pattern (`strategies/rsi_long_short.py`, `strategies/mean_reversion_atr.py`).

**Reference spec:** `docs/superpowers/specs/2026-05-18-ml-supertrend-strategy-design.md`

---

## File Structure

- **Create** `strategies/ml_supertrend.py` — `MLSupertrendParams` dataclass, four helper functions, `MLSupertrendStrategy` class. Single responsibility: this one strategy.
- **Modify** `backtester/strategies/registry.py` — add one import and one `register_strategy(...)` line in the curated block.
- **Create** `configs/backtests/ml_supertrend_spy.yaml` — TradingView-parity SPY backtest config.
- **Create** `tests/unit/test_strategy_ml_supertrend.py` — all unit tests for the strategy and its helpers.

---

## Task 1: Params dataclass + strategy skeleton

**Files:**
- Create: `strategies/ml_supertrend.py`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_strategy_ml_supertrend.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def test_params_type_and_defaults():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    assert MLSupertrendStrategy.params_type() is MLSupertrendParams
    p = MLSupertrendParams()
    assert p.signal_mode == "reversal"
    assert p.require_new_extreme is True
    assert p.min_bars_between_signals == 10
    assert p.sensitivity == 30
    assert p.atr_period == 24
    assert p.multiplier == 1.4
    assert p.source_type == "hlcc4"
    assert p.use_atr is True
    assert p.enable_rsi is True
    assert p.rsi_len == 14
    assert p.rsi_lookback_top == 50
    assert p.rsi_lookback_bot == 50
    assert p.rsi_top == 70
    assert p.rsi_bot == 30
    assert p.vol_lookback == 3
    assert p.vol_multiplier == 1.2
    assert p.require_vol_spike is False
    assert p.enable_major_levels_only is False
    assert p.major_level_threshold == 4.5
    assert p.size == 1.0


def test_strategy_identity_and_warmup():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    s = MLSupertrendStrategy()
    assert s.strategy_id == "ml_supertrend"
    assert s.timeframe == "1d"
    # warmup = max(atr_period, sensitivity, rsi_len, vol_lookback) + 1
    assert s.warmup_bars(MLSupertrendParams()) == 31
    assert s.warmup_bars(MLSupertrendParams(atr_period=60, sensitivity=10)) == 61
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategies.ml_supertrend'`

- [ ] **Step 3: Write minimal implementation**

Create `strategies/ml_supertrend.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MLSupertrendParams:
    # Group 1 — signal mode
    signal_mode: str = "reversal"          # "reversal" | "breakout"
    require_new_extreme: bool = True
    min_bars_between_signals: int = 10
    # Group 2 — volatility envelope
    sensitivity: int = 30
    atr_period: int = 24
    multiplier: float = 1.4
    source_type: str = "hlcc4"
    use_atr: bool = True
    # Group 3 — momentum filter
    enable_rsi: bool = True
    rsi_len: int = 14
    rsi_lookback_top: int = 50
    rsi_lookback_bot: int = 50
    rsi_top: int = 70
    rsi_bot: int = 30
    # Group 4 — flow analysis
    vol_lookback: int = 3
    vol_multiplier: float = 1.2
    require_vol_spike: bool = False
    # Group 5 — signal quality
    enable_major_levels_only: bool = False
    major_level_threshold: float = 4.5
    # Position sizing
    size: float = 1.0


class MLSupertrendStrategy(BaseStrategy[MLSupertrendParams]):
    """
    Purpose:
        SuperTrend + new-extreme reversal/breakout strategy, ported from the
        signal core of the "Machine Learning Supertrend [Aslan]" Pine Script.

        NOTE: the Pine Script's adaptive "ML" self-tuning engine is intentionally
        NOT ported. Parameters are static — tune them with the suite's
        grid-search / walk-forward optimization, not an in-sample self-tuner.

    Inputs:
        OHLCV dataframe with datetime index and lowercase columns:
        open, high, low, close, volume.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} (stop-and-reverse held position)
        and `size`.

    Requires:
        ExecutionConfig.allow_short = True. The stop-and-reverse model goes
        short on every Sell; without allow_short the simulator raises
        ShortNotAllowedError on the first -1.
    """

    strategy_id = "ml_supertrend"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return MLSupertrendParams

    def warmup_bars(self, params: MLSupertrendParams) -> int:
        return max(
            params.atr_period,
            params.sensitivity,
            params.rsi_len,
            params.vol_lookback,
        ) + 1

    def indicators(self, data: pd.DataFrame, params: MLSupertrendParams) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MLSupertrendParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: add ml_supertrend params and strategy skeleton"
```

---

## Task 2: Source resolution + smoothed true range helpers

**Files:**
- Modify: `strategies/ml_supertrend.py`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def _ohlcv(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    steps = rng.normal(0.0, 1.0, n).cumsum()
    close = 100.0 + steps
    high = close + np.abs(rng.normal(0.0, 0.5, n)) + 0.5
    low = close - np.abs(rng.normal(0.0, 0.5, n)) - 0.5
    open_ = close + rng.normal(0.0, 0.3, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(800_000, 1_200_000, n).astype(float)},
        index=idx,
    )


def test_resolve_source_variants():
    from strategies.ml_supertrend import _resolve_source

    data = _ohlcv()
    assert _resolve_source(data, "close").equals(data["close"])
    assert _resolve_source(data, "high").equals(data["high"])
    hl2 = _resolve_source(data, "hl2")
    pd.testing.assert_series_equal(hl2, (data["high"] + data["low"]) / 2.0, check_names=False)
    hlcc4 = _resolve_source(data, "hlcc4")
    expected = (data["high"] + data["low"] + data["close"] + data["close"]) / 4.0
    pd.testing.assert_series_equal(hlcc4, expected, check_names=False)
    with pytest.raises(ValueError):
        _resolve_source(data, "nonsense")


def test_smoothed_tr_rma_vs_ema():
    from strategies.ml_supertrend import _smoothed_tr

    data = _ohlcv()
    rma = _smoothed_tr(data, period=14, use_atr=True)
    ema = _smoothed_tr(data, period=14, use_atr=False)
    # First bar of both equals high - low of bar 0 (TR seed).
    assert rma.iloc[0] == pytest.approx(data["high"].iloc[0] - data["low"].iloc[0])
    assert ema.iloc[0] == pytest.approx(data["high"].iloc[0] - data["low"].iloc[0])
    # No NaN (ewm with adjust=False seeds from bar 0).
    assert not rma.isna().any()
    assert not ema.isna().any()
    # RMA (alpha=1/14) and EMA (alpha=2/15) smooth differently.
    assert not np.allclose(rma.to_numpy(), ema.to_numpy())
    # All positive.
    assert (rma > 0).all() and (ema > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_resolve_source_variants tests/unit/test_strategy_ml_supertrend.py::test_smoothed_tr_rma_vs_ema -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_source'`

- [ ] **Step 3: Write minimal implementation**

In `strategies/ml_supertrend.py`, insert these two functions immediately after the imports and before the `MLSupertrendParams` dataclass:

```python
def _resolve_source(data: pd.DataFrame, source_type: str) -> pd.Series:
    """Resolve the Pine `sourceType` input to a price series."""
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    if source_type == "open":
        return o
    if source_type == "high":
        return h
    if source_type == "low":
        return l
    if source_type == "close":
        return c
    if source_type == "hl2":
        return (h + l) / 2.0
    if source_type == "hlc3":
        return (h + l + c) / 3.0
    if source_type == "ohlc4":
        return (o + h + l + c) / 4.0
    if source_type == "hlcc4":
        return (h + l + c + c) / 4.0
    raise ValueError(f"Unknown source_type: {source_type!r}")


def _smoothed_tr(data: pd.DataFrame, period: int, use_atr: bool) -> pd.Series:
    """True Range smoothed by Wilder RMA (use_atr=True) or EMA (use_atr=False).

    Mirrors the Pine `rma_var`/`ema_var` of `ta.tr`: TR[0] seeds with
    high[0]-low[0], and the recurrence is seeded from that first value
    (ewm adjust=False), so there are no NaN values.
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    alpha = (1.0 / period) if use_atr else (2.0 / (period + 1.0))
    return tr.ewm(alpha=alpha, adjust=False).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: add source resolution and smoothed TR helpers to ml_supertrend"
```

---

## Task 3: SuperTrend trend-state helper

**Files:**
- Modify: `strategies/ml_supertrend.py`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def test_supertrend_trend_range_and_start():
    from strategies.ml_supertrend import _supertrend_trend

    n = 60
    close = np.full(n, 100.0)
    src = close.copy()
    atr = np.full(n, 2.0)
    trend = _supertrend_trend(src, atr, close, multiplier=1.0)
    assert trend.shape == (n,)
    assert set(np.unique(trend)).issubset({-1, 1})
    # On a flat series price never crosses a band, trend stays at its +1 start.
    assert trend[0] == 1
    assert (trend == 1).all()


def test_supertrend_trend_flips_down_then_up():
    from strategies.ml_supertrend import _supertrend_trend

    # Rise for 30 bars, then a sharp sustained fall, then a sharp rise.
    up = np.linspace(100.0, 130.0, 30)
    down = np.linspace(130.0, 70.0, 30)
    up2 = np.linspace(70.0, 110.0, 30)
    close = np.concatenate([up, down, up2])
    src = close.copy()
    atr = np.full(close.shape[0], 2.0)
    trend = _supertrend_trend(src, atr, close, multiplier=1.0)
    assert trend[0] == 1
    # Somewhere in the decline the trend must flip to -1.
    assert (trend[30:60] == -1).any(), "expected a downtrend during the fall"
    # And flip back to +1 during the final rise.
    assert (trend[60:] == 1).any(), "expected an uptrend during the recovery"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_supertrend_trend_range_and_start tests/unit/test_strategy_ml_supertrend.py::test_supertrend_trend_flips_down_then_up -v`
Expected: FAIL — `ImportError: cannot import name '_supertrend_trend'`

- [ ] **Step 3: Write minimal implementation**

In `strategies/ml_supertrend.py`, insert this function immediately after `_smoothed_tr`:

```python
def _supertrend_trend(
    src: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """SuperTrend trend state in {+1, -1}, faithful to Pine `getSupertrend_var`.

    `support` is the band below price, `resistance` the band above. Bands
    ratchet using the *previous* close vs the *previous* band; the trend flips
    using the *current* close vs the *previous* band. Trend starts at +1.

    (The Pine code names these `upper`/`lower` with swapped meanings — see spec
    section 6.2. Behaviour here is identical.)
    """
    src = np.asarray(src, dtype=float)
    atr = np.asarray(atr, dtype=float)
    close = np.asarray(close, dtype=float)
    n = src.shape[0]
    support = src - multiplier * atr
    resistance = src + multiplier * atr
    trend = np.ones(n, dtype=np.int64)
    for i in range(1, n):
        if close[i - 1] > support[i - 1]:
            support[i] = max(support[i], support[i - 1])
        if close[i - 1] < resistance[i - 1]:
            resistance[i] = min(resistance[i], resistance[i - 1])
        if trend[i - 1] == -1 and close[i] > resistance[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < support[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return trend
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: add SuperTrend trend-state helper to ml_supertrend"
```

---

## Task 4: Wilder RSI helper

**Files:**
- Modify: `strategies/ml_supertrend.py`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def test_wilder_rsi_range_and_warmup():
    from strategies.ml_supertrend import _wilder_rsi

    data = _ohlcv(n=60, seed=3)
    rsi = _wilder_rsi(data["close"], length=14)
    valid = rsi.dropna()
    # First `length` values are NaN (min_periods=length).
    assert rsi.iloc[:13].isna().all()
    assert not rsi.iloc[14:].isna().any()
    # RSI is bounded to [0, 100].
    assert (valid >= 0).all() and (valid <= 100).all()


def test_wilder_rsi_all_gains_is_100():
    from strategies.ml_supertrend import _wilder_rsi

    # Monotonically rising close -> no losses -> RSI saturates at 100.
    close = pd.Series(np.linspace(100.0, 160.0, 40))
    rsi = _wilder_rsi(close, length=14)
    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_wilder_rsi_range_and_warmup tests/unit/test_strategy_ml_supertrend.py::test_wilder_rsi_all_gains_is_100 -v`
Expected: FAIL — `ImportError: cannot import name '_wilder_rsi'`

- [ ] **Step 3: Write minimal implementation**

In `strategies/ml_supertrend.py`, insert this function immediately after `_supertrend_trend`:

```python
def _wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder RSI (alpha = 1/length), matching Pine `ta.rsi`. Wilder smoothing
    as in `rsi_long_short.py`; the first `length` values are NaN. A window with
    no down moves yields RSI = 100 (the Pine convention) rather than NaN, so the
    RSI filter is not silently disabled during strong rallies."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 with gains present -> rs is +inf -> RSI = 100. The replace()
    # above turned that into NaN; restore 100 explicitly.
    no_loss = (avg_loss == 0.0) & (avg_gain > 0.0)
    return rsi.mask(no_loss, 100.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: add Wilder RSI helper to ml_supertrend"
```

---

## Task 5: `indicators()` assembly

**Files:**
- Modify: `strategies/ml_supertrend.py:indicators`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
_IND_COLUMNS = {
    "atr", "st_trend", "rsi", "roll_high", "roll_low",
    "is_new_high", "is_new_low", "rsi_cold", "rsi_hot",
    "vol_surge", "sig_high", "sig_low",
}


def test_indicators_produces_all_columns():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=120, seed=5)
    ind = MLSupertrendStrategy().indicators(data, MLSupertrendParams())
    assert _IND_COLUMNS.issubset(set(ind.columns))
    assert len(ind) == len(data)
    assert set(np.unique(ind["st_trend"].to_numpy())).issubset({-1, 1})
    assert ind["atr"].dropna().gt(0).all()
    for col in ("is_new_high", "is_new_low", "rsi_cold", "rsi_hot",
                "vol_surge", "sig_high", "sig_low"):
        assert ind[col].dtype == bool


def test_indicators_rsi_filters_off_when_disabled():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=120, seed=6)
    ind = MLSupertrendStrategy().indicators(
        data, MLSupertrendParams(enable_rsi=False)
    )
    # With RSI disabled both filter columns are constant True.
    assert ind["rsi_cold"].all()
    assert ind["rsi_hot"].all()


def test_indicators_major_levels_filter_is_subset():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _ohlcv(n=200, seed=7)
    strat = MLSupertrendStrategy()
    base = strat.indicators(data, MLSupertrendParams(enable_major_levels_only=False))
    major = strat.indicators(data, MLSupertrendParams(enable_major_levels_only=True))
    # The key-levels filter can only remove fresh extremes, never add them.
    assert (major["sig_high"] <= base["sig_high"]).all()
    assert (major["sig_low"] <= base["sig_low"]).all()
    # When the filter is off, sig_* equals is_new_* exactly.
    assert base["sig_high"].equals(base["is_new_high"])
    assert base["sig_low"].equals(base["is_new_low"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_indicators_produces_all_columns -v`
Expected: FAIL — `AssertionError` (the skeleton `indicators()` returns an empty frame)

- [ ] **Step 3: Write minimal implementation**

In `strategies/ml_supertrend.py`, replace the entire `indicators` method body with:

```python
    def indicators(self, data: pd.DataFrame, params: MLSupertrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]

        src = _resolve_source(data, params.source_type)
        atr = _smoothed_tr(data, params.atr_period, params.use_atr)
        out["atr"] = atr
        out["st_trend"] = _supertrend_trend(
            src.to_numpy(), atr.to_numpy(), close.to_numpy(), params.multiplier
        )
        out["rsi"] = _wilder_rsi(close, params.rsi_len)

        # Rolling extremes over the sensitivity window.
        roll_high = data["high"].rolling(params.sensitivity).max()
        roll_low = data["low"].rolling(params.sensitivity).min()
        out["roll_high"] = roll_high
        out["roll_low"] = roll_low

        # Fresh-pivot detection: the rolling extreme changed vs `lookback` bars
        # ago AND close pushed past that prior extreme.
        lookback = max(1, int(round(params.sensitivity / 10.0)))
        prev_high = roll_high.shift(lookback)
        prev_low = roll_low.shift(lookback)
        out["is_new_high"] = (
            roll_high.notna() & prev_high.notna()
            & (roll_high != prev_high) & (close > prev_high)
        )
        out["is_new_low"] = (
            roll_low.notna() & prev_low.notna()
            & (roll_low != prev_low) & (close < prev_low)
        )

        # RSI hot/cold memory: was RSI past the threshold within the lookback?
        if params.enable_rsi:
            cold = out["rsi"] < params.rsi_bot
            hot = out["rsi"] > params.rsi_top
            out["rsi_cold"] = (
                cold.rolling(params.rsi_lookback_bot, min_periods=1).max()
                .fillna(0.0).astype(bool)
            )
            out["rsi_hot"] = (
                hot.rolling(params.rsi_lookback_top, min_periods=1).max()
                .fillna(0.0).astype(bool)
            )
        else:
            out["rsi_cold"] = pd.Series(True, index=data.index)
            out["rsi_hot"] = pd.Series(True, index=data.index)

        # Volume surge.
        vol_avg = data["volume"].rolling(params.vol_lookback).mean()
        out["vol_surge"] = (data["volume"] > params.vol_multiplier * vol_avg).fillna(False)

        # Key-levels filter: only the biggest structural pivots survive.
        if params.enable_major_levels_only:
            depth = atr * params.major_level_threshold
            out["sig_high"] = out["is_new_high"] & ((data["high"] - roll_low) > depth)
            out["sig_low"] = out["is_new_low"] & ((roll_high - data["low"]) > depth)
        else:
            out["sig_high"] = out["is_new_high"]
            out["sig_low"] = out["is_new_low"]

        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: implement ml_supertrend indicators()"
```

---

## Task 6: `generate_signals()` — reversal mode

**Files:**
- Modify: `strategies/ml_supertrend.py:generate_signals`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def _swinging_ohlcv(n=300, seed=11):
    """Trend-swinging series: repeated up/down legs so SuperTrend flips
    multiple times and fresh highs/lows occur in both directions."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.empty(n)
    p = 100.0
    for i in range(n):
        leg = (i // 25) % 2          # alternate 25-bar up / down legs
        drift = 0.6 if leg == 0 else -0.6
        p = max(5.0, p + drift + rng.normal(0.0, 0.4))
        close[i] = p
    high = close + np.abs(rng.normal(0.0, 0.4, n)) + 0.3
    low = close - np.abs(rng.normal(0.0, 0.4, n)) - 0.3
    open_ = close + rng.normal(0.0, 0.2, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(800_000, 1_200_000, n).astype(float)},
        index=idx,
    )


def _run(strat, data, params):
    ind = strat.indicators(data, params)
    ctx = StrategyContext(symbol="SYN", timeframe="1d",
                          warmup_bars=strat.warmup_bars(params))
    return strat.generate_signals(data, ind, ctx, params).data["signal"]


def test_reversal_signal_values_and_first_bar_flat():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data, MLSupertrendParams(signal_mode="reversal"))
    assert set(sigs.unique()).issubset({-1, 0, 1})
    assert sigs.iloc[0] == 0          # shift(1) leaves the first bar flat
    assert len(sigs) == len(data)


def test_reversal_no_signal_in_warmup():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    strat = MLSupertrendStrategy()
    params = MLSupertrendParams(signal_mode="reversal")
    sigs = _run(strat, data, params)
    # Every bar up to and including warmup index is flat.
    assert (sigs.iloc[: strat.warmup_bars(params) + 1] == 0).all()


def test_reversal_is_stop_and_reverse():
    """Once trading starts the position is never flat again, and consecutive
    distinct non-zero values alternate +1 / -1 (stop-and-reverse)."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data,
                MLSupertrendParams(signal_mode="reversal")).to_numpy()
    nz = sigs[sigs != 0]
    assert nz.size > 0, "expected at least one signal on a swinging series"
    # Collapse runs of equal values; the distinct sequence must alternate.
    collapsed = nz[np.insert(np.diff(nz) != 0, 0, True)]
    assert np.all(np.abs(np.diff(collapsed)) == 2), "non-zero signal must alternate +1/-1"
    # After the first signal there is no return to flat.
    first = np.argmax(sigs != 0)
    assert np.all(sigs[first:] != 0)


def test_reversal_signal_spacing_is_honoured():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    params = MLSupertrendParams(signal_mode="reversal", min_bars_between_signals=20)
    sigs = _run(MLSupertrendStrategy(), data, params).to_numpy()
    # A "new signal bar" is where the held position changes value.
    change_idx = np.where(np.diff(sigs) != 0)[0] + 1
    # Drop the initial 0 -> first-signal transition is still a real signal;
    # gaps between successive signal bars must be >= min_bars_between_signals.
    gaps = np.diff(change_idx)
    assert np.all(gaps >= params.min_bars_between_signals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_reversal_is_stop_and_reverse -v`
Expected: FAIL — the skeleton `generate_signals()` emits all zeros, so `nz.size > 0` fails.

- [ ] **Step 3: Write minimal implementation**

In `strategies/ml_supertrend.py`, replace the entire `generate_signals` method body with:

```python
    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MLSupertrendParams,
    ) -> SignalFrame:
        n = len(data)
        st = indicators["st_trend"].to_numpy()
        sig_high = indicators["sig_high"].to_numpy(dtype=bool)
        sig_low = indicators["sig_low"].to_numpy(dtype=bool)
        rsi_cold = indicators["rsi_cold"].to_numpy(dtype=bool)
        rsi_hot = indicators["rsi_hot"].to_numpy(dtype=bool)
        vol_surge = indicators["vol_surge"].to_numpy(dtype=bool)

        warmup = self.warmup_bars(params)

        position = np.zeros(n, dtype=np.int64)
        held = 0
        last_signal_bar = 0          # matches Pine `var int lastSignalBar = 0`
        top_flag = 0
        bot_flag = 0

        for i in range(n):
            buy = False
            sell = False
            prev_top = top_flag
            prev_bot = bot_flag
            spaced = (i - last_signal_bar) >= params.min_bars_between_signals

            # Reversal-mode flag latches update whenever spacing allows
            # (faithful to Pine: flags live inside `if enableReversal and canSignal`).
            if params.signal_mode == "reversal" and spaced:
                if st[i] == -1:
                    top_flag = 0
                elif sig_high[i] and st[i] == 1:
                    top_flag = 1
                if st[i] == 1:
                    bot_flag = 0
                elif sig_low[i] and st[i] == -1:
                    bot_flag = 1

            if i >= warmup and spaced:
                buy_filters = rsi_cold[i] and (
                    not params.require_vol_spike or vol_surge[i]
                )
                sell_filters = rsi_hot[i] and (
                    not params.require_vol_spike or vol_surge[i]
                )
                if params.signal_mode == "reversal":
                    flip_down = i > 0 and st[i - 1] == 1 and st[i] == -1
                    flip_up = i > 0 and st[i - 1] == -1 and st[i] == 1
                    rev_sell = (prev_top == 1 and top_flag == 0) or (
                        not params.require_new_extreme and flip_down
                    )
                    rev_buy = (prev_bot == 1 and bot_flag == 0) or (
                        not params.require_new_extreme and flip_up
                    )
                    if rev_sell and sell_filters:
                        sell = True
                    elif rev_buy and buy_filters:
                        buy = True
                else:  # breakout
                    if sig_high[i] and st[i] == 1 and sell_filters:
                        sell = True
                    elif sig_low[i] and st[i] == -1 and buy_filters:
                        buy = True

            if buy:
                held = 1
                last_signal_bar = i
            elif sell:
                held = -1
                last_signal_bar = i
            position[i] = held

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(position, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add strategies/ml_supertrend.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: implement ml_supertrend generate_signals() with reversal mode"
```

---

## Task 7: `generate_signals()` — breakout mode coverage

The breakout branch is already implemented in Task 6. This task adds the breakout-mode and mode-difference tests from spec section 8.1.

**Files:**
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def test_breakout_emits_signals_and_is_stop_and_reverse():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    sigs = _run(MLSupertrendStrategy(), data,
                MLSupertrendParams(signal_mode="breakout")).to_numpy()
    assert set(np.unique(sigs)).issubset({-1, 0, 1})
    nz = sigs[sigs != 0]
    assert nz.size > 0, "expected at least one breakout signal"
    collapsed = nz[np.insert(np.diff(nz) != 0, 0, True)]
    assert np.all(np.abs(np.diff(collapsed)) == 2), "breakout signal must alternate +1/-1"


def test_reversal_and_breakout_differ_on_same_series():
    """Spec section 8.1: on the same series the two modes fire at different
    times (Breakout fires on the fresh-extreme bar, Reversal later on the
    confirmed trend flip), so the held-position series are not identical."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv()
    strat = MLSupertrendStrategy()
    rev = _run(strat, data, MLSupertrendParams(signal_mode="reversal"))
    brk = _run(strat, data, MLSupertrendParams(signal_mode="breakout"))
    assert not rev.equals(brk), "reversal and breakout must not produce identical signals"


def test_require_vol_spike_blocks_signals_without_surges():
    """With constant volume there is never a surge, so require_vol_spike=True
    blocks every signal. The same series without the gate still trades."""
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    data = _swinging_ohlcv().copy()
    data["volume"] = 1_000_000.0          # constant -> vol_surge is always False
    strat = MLSupertrendStrategy()

    gated = _run(strat, data,
                 MLSupertrendParams(signal_mode="breakout", require_vol_spike=True))
    assert (gated == 0).all(), "no surge -> require_vol_spike blocks every signal"

    ungated = _run(strat, data,
                   MLSupertrendParams(signal_mode="breakout", require_vol_spike=False))
    assert (ungated != 0).any(), "without the vol gate the series still trades"
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_breakout_emits_signals_and_is_stop_and_reverse tests/unit/test_strategy_ml_supertrend.py::test_reversal_and_breakout_differ_on_same_series tests/unit/test_strategy_ml_supertrend.py::test_require_vol_spike_blocks_signals_without_surges -v`
Expected: PASS — the breakout branch was implemented in Task 6, so these new tests pass immediately. (If any fails, the breakout branch in `generate_signals()` has a bug — fix it before committing.)

- [ ] **Step 3: Run the whole strategy test file**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (18 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_strategy_ml_supertrend.py
git commit -m "test: cover ml_supertrend breakout mode and mode divergence"
```

---

## Task 8: Register the strategy

**Files:**
- Modify: `backtester/strategies/registry.py`
- Test: `tests/unit/test_strategy_ml_supertrend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_strategy_ml_supertrend.py`:

```python
def test_ml_supertrend_is_registered():
    from backtester.strategies.registry import get_strategy_class
    from strategies.ml_supertrend import MLSupertrendStrategy

    assert get_strategy_class("ml_supertrend") is MLSupertrendStrategy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py::test_ml_supertrend_is_registered -v`
Expected: FAIL — `KeyError: "Strategy 'ml_supertrend' is not registered."`

- [ ] **Step 3: Write minimal implementation**

In `backtester/strategies/registry.py`, in the curated-strategy block:

Add this import line after the existing `from strategies.mean_reversion_atr import ...` line:

```python
from strategies.ml_supertrend import MLSupertrendStrategy  # noqa: E402
```

Add this registration line after the existing `register_strategy(MeanReversionAtrStrategy)` line:

```python
register_strategy(MLSupertrendStrategy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py -v`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add backtester/strategies/registry.py tests/unit/test_strategy_ml_supertrend.py
git commit -m "feat: register ml_supertrend strategy"
```

---

## Task 9: TradingView-parity SPY config + end-to-end smoke run

**Files:**
- Create: `configs/backtests/ml_supertrend_spy.yaml`

- [ ] **Step 1: Create the config**

Create `configs/backtests/ml_supertrend_spy.yaml`:

```yaml
# TradingView-parity baseline. Every value under `strategy_params` is copied
# verbatim from the "Machine Learning Supertrend [Aslan]" Pine Script input
# defaults. A tuned config (grid-search / WFO output) must be named distinctly
# so a reader can always tell a TV-parity run from an optimized one.
run_name: ml_supertrend_spy
strategy: ml_supertrend
strategy_params:
  signal_mode: "reversal"
  require_new_extreme: true
  min_bars_between_signals: 10
  sensitivity: 30
  atr_period: 24
  multiplier: 1.4
  source_type: "hlcc4"
  use_atr: true
  enable_rsi: true
  rsi_len: 14
  rsi_lookback_top: 50
  rsi_lookback_bot: 50
  rsi_top: 70
  rsi_bot: 30
  vol_lookback: 3
  vol_multiplier: 1.2
  require_vol_spike: false
  enable_major_levels_only: false
  major_level_threshold: 4.5
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

- [ ] **Step 2: Run the backtest end-to-end**

Run: `python -m backtester.runners.run_backtest --config configs/backtests/ml_supertrend_spy.yaml`
Expected: the run completes with no error and prints an output bundle path under `output/runs/`. (`data/raw/SPY.csv` already exists in the repo.)

- [ ] **Step 3: Verify the artifact bundle**

Run: `python -c "import json,glob,os; d=sorted(glob.glob('output/runs/*ml_supertrend_spy'))[-1]; print(d); s=json.load(open(os.path.join(d,'summary.json'))); print(sorted(s.keys()))"`
Expected: prints the run directory and the summary keys (sharpe, drawdown, win rate, etc.) — confirms `summary.json`, and therefore the full bundle, was written.

- [ ] **Step 4: Run the full unit + strategy test suite**

Run: `python -m pytest tests/unit/test_strategy_ml_supertrend.py tests/unit/test_strategy_registry.py -v`
Expected: PASS — the new strategy and the registry test both green.

- [ ] **Step 5: Commit**

```bash
git add configs/backtests/ml_supertrend_spy.yaml
git commit -m "feat: add TradingView-parity SPY backtest config for ml_supertrend"
```

---

## Self-Review Notes

**Spec coverage check (against `2026-05-18-ml-supertrend-strategy-design.md`):**

- §1–2 (purpose, drop the ML engine) — Task 1 docstring states the adaptive engine is not ported; no group ⑥–⑫ parameters exist.
- §3 (vectorized interface, identity, warmup) — Tasks 1, 5, 6.
- §4 (`MLSupertrendParams`, all fields + defaults) — Task 1, asserted by `test_params_type_and_defaults`.
- §5 (stop-and-reverse, `allow_short`) — Task 6 loop logic; `test_reversal_is_stop_and_reverse`; Task 9 config sets `allow_short: true`.
- §6.1 (every indicator column) — Task 5; `_IND_COLUMNS` assertion.
- §6.2 (SuperTrend recurrence) — Task 3.
- §6.3 (sequential loop: spacing, reversal flags, stop-and-reverse, breakout) — Task 6.
- §6.4 (warmup masking, single `shift(1)`) — Task 6; `test_reversal_no_signal_in_warmup`, `test_reversal_signal_values_and_first_bar_flat`.
- §7 (4 deliverables) — Tasks 1/5/6 (strategy file), 8 (registry), 9 (config), tests throughout.
- §8 (testing strategy) + §8.1 (reversal-vs-breakout) — Tasks 2–9 tests; `test_reversal_and_breakout_differ_on_same_series`.
- §9 (contrarian polarity, recursive loop, shorting) — polarity is baked into Task 6 (`sig_high` → Sell); recursive loop in Task 3; `allow_short` in Task 9.

**Type consistency:** helper names (`_resolve_source`, `_smoothed_tr`, `_supertrend_trend`, `_wilder_rsi`), the indicator column set (`_IND_COLUMNS`), and `MLSupertrendParams` field names are identical across every task that references them.

**Note on spec §8.1:** the spec describes asserting an exact bar-index gap between modes. The plan instead asserts the two modes produce non-identical signal series (`test_reversal_and_breakout_differ_on_same_series`) plus per-mode signal presence — this captures the same divergence without a fixture pinned to exact ATR arithmetic, which would be fragile. The executor may tighten it to an exact-index assertion if a deterministic fixture is built.
