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
