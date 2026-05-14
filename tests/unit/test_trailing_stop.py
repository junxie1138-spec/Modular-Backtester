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
