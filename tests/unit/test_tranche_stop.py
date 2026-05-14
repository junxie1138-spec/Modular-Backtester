import math

import pandas as pd
import pytest


# ---- helpers ----

def _atr_series(values, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="atr")


def _make_state(*, hard=1.75, runner=2.5, breakeven_floor=True, atr_series=None):
    from backtester.engine.tranche_stop import TrancheStopState
    if atr_series is None:
        atr_series = _atr_series([2.0] * 20)
    return TrancheStopState(
        hard_stop_atr_mult=hard,
        runner_atr_mult=runner,
        breakeven_floor=breakeven_floor,
        atr_series=atr_series,
    )


# ---- 5 state-machine tests ----

def test_disarmed_by_default():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    assert ts.phase is TSPhase.DISARMED


def test_reset_sets_hard_phase_and_snapshots_entry():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=5)
    assert ts.phase is TSPhase.HARD
    assert ts.entry_price == 100.0
    assert ts.entry_bar_idx == 5
    assert ts.atr_at_entry == 2.0
    assert ts.peak_close == 100.0
    assert ts.trough_close == 100.0


def test_promote_to_runner_keeps_peak_close():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    # Simulate a few up-bars in HARD phase.
    for c in [101.0, 102.5, 104.0]:
        ts.update(pd.Series({"high": c + 1, "low": c - 1, "close": c}))
    assert ts.peak_close == 104.0
    ts.promote_to_runner()
    assert ts.phase is TSPhase.RUNNER
    assert ts.peak_close == 104.0  # peak persists across promotion


def test_promote_to_runner_idempotent():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    ts.promote_to_runner()  # second call no-op
    assert ts.phase is TSPhase.RUNNER


def test_disarm_clears_peak_close_and_phase():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.update(pd.Series({"high": 105, "low": 99, "close": 103}))
    ts.disarm()
    assert ts.phase is TSPhase.DISARMED
    assert ts.peak_close == 0.0
    assert math.isinf(ts.trough_close)


# ---- 5 stop_price tests ----

def test_hard_stop_uses_atr_at_entry_not_current_atr():
    """HARD-phase stop is fixed at entry; current ATR is irrelevant."""
    # ATR rises across the holding period, but HARD stop uses the entry ATR.
    atr = _atr_series([2.0, 2.0, 2.0, 10.0, 10.0])
    ts = _make_state(hard=1.75, runner=2.5, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    # ATR at entry = 2.0 -> hard stop = 100 - 1.75*2 = 96.5
    assert ts.stop_price(sign=+1, bar_idx=3) == pytest.approx(96.5)
    # Even though atr_series[3] = 10.0, the stop does not move.


def test_hard_stop_does_not_trail():
    atr = _atr_series([2.0] * 10)
    ts = _make_state(atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    # Price ratchets up, but the stop should remain at entry-based level.
    for i, c in enumerate([101.0, 103.0, 105.0], start=1):
        ts.update(pd.Series({"high": c + 0.5, "low": c - 0.5, "close": c}))
    # HARD stop = 100 - 1.75*2 = 96.5; does not move even though peak_close = 105.
    assert ts.stop_price(sign=+1, bar_idx=3) == pytest.approx(96.5)


def test_runner_trail_uses_peak_close_not_peak_high():
    """Critical: intrabar wicks (high) DO NOT move the runner trail."""
    atr = _atr_series([2.0] * 10)
    ts = _make_state(runner=2.5, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Update with bars whose HIGH is well above the CLOSE.
    ts.update(pd.Series({"high": 120.0, "low": 99.0, "close": 105.0}))
    # peak_close is 105.0, not 120.0.
    # Runner stop = 105 - 2.5*2 = 100.0; with breakeven floor active -> max(100, 100) = 100.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(100.0)


def test_runner_trail_clipped_by_breakeven_floor():
    """When raw trail is below entry, breakeven_floor clamps it up to entry_price."""
    atr = _atr_series([5.0] * 10)
    ts = _make_state(runner=2.5, breakeven_floor=True, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Price didn't move much; peak_close ~ 100.
    ts.update(pd.Series({"high": 101.0, "low": 99.0, "close": 100.5}))
    # Raw trail = 100.5 - 2.5*5 = 88.0; floored at entry_price = 100.0.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(100.0)


def test_runner_trail_no_floor_when_disabled():
    atr = _atr_series([5.0] * 10)
    ts = _make_state(runner=2.5, breakeven_floor=False, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    ts.update(pd.Series({"high": 101.0, "low": 99.0, "close": 100.5}))
    # Raw trail = 100.5 - 12.5 = 88.0; no floor.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(88.0)
