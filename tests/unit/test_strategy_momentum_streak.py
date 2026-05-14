from __future__ import annotations

import numpy as np
import pandas as pd

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
    volumes = [1_000_000] * 3 + [5_000_000, 5_000_000, 6_000_000] + [1_000_000]
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
    volumes = [1_000_000] * 3 + [5_000_000, 5_000_000, 6_000_000] + [1_000_000] * 4
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
    volumes = [1_000_000] * 3 + [5_000_000, 5_000_000, 6_000_000] + [1_000_000]
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
    volumes = [1_000_000] * 3 + [5_000_000, 5_000_000, 6_000_000] + [1_000_000] + [5_000_000, 5_000_000, 6_000_000] + [1_000_000]
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
