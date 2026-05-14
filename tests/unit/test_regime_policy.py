import pandas as pd
import pytest


def _spy_series(values, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"close": values}, index=idx)


def _vix_series(values):
    return _spy_series(values)


def test_spy_ema_trips_below_threshold():
    from backtester.engine.regime import SpyEmaGate
    gate = SpyEmaGate(ema_lookback=3, trip_pct=-0.02, resume_pct=0.02)
    spy = _spy_series([100, 100, 100, 100, 80])
    for i in range(len(spy)):
        gate.update(bar_idx=i, spy_close=spy["close"], spy_ema=spy["close"].ewm(span=3, adjust=False).mean())
    assert gate.tripped is True


def test_spy_ema_resumes_above_threshold_hysteresis():
    from backtester.engine.regime import SpyEmaGate
    gate = SpyEmaGate(ema_lookback=3, trip_pct=-0.02, resume_pct=0.02)
    closes = [100, 100, 100, 100, 80, 95, 98, 110]
    spy = _spy_series(closes)
    ema = spy["close"].ewm(span=3, adjust=False).mean()
    for i in range(len(spy)):
        gate.update(bar_idx=i, spy_close=spy["close"], spy_ema=ema)
    assert gate.tripped is False


def test_vix_requires_two_consecutive_above_30():
    from backtester.engine.regime import VixGate
    gate = VixGate(trip_threshold=30, trip_consec=2, resume_threshold=25, resume_consec=3)
    vix = _vix_series([20, 31, 20, 31, 32])
    for i in range(len(vix)):
        gate.update(bar_idx=i, vix_close=vix["close"])
    assert gate.tripped is True


def test_vix_resume_requires_three_consecutive_below_25():
    from backtester.engine.regime import VixGate
    gate = VixGate(trip_threshold=30, trip_consec=2, resume_threshold=25, resume_consec=3)
    vix = _vix_series([31, 32, 20, 20, 20])
    for i in range(len(vix)):
        gate.update(bar_idx=i, vix_close=vix["close"])
    assert gate.tripped is False


def test_circuit_breaker_trips_on_minus_5_pct_rolling_20d():
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    pnls = [-1000.0] * 5 + [0.0] * 5
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    assert gate.tripped_history[4] is True


def test_circuit_breaker_resumes_after_pause_days():
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    pnls = [-1000.0] * 5 + [0.0, 0.0, 0.0, 0.0]
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    # Tripped at bar 4 (idx 4); pause_days=2 means resume at bar 4 + 2 + 1 = 7.
    assert gate.tripped_history[7] is False


def test_circuit_breaker_resumes_at_full_size_not_phased():
    """PRD literal: re-entry on day 11 is at full size, no phased ramp."""
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    pnls = [-1000.0] * 5 + [0.0] * 5
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    assert not gate.tripped
    assert not hasattr(gate, "phased_size_mult")


def test_book_flat_is_disjunction_across_gates():
    from backtester.engine.regime import RegimePolicy
    policy = RegimePolicy.from_disabled()
    policy.spy_ema.tripped = True
    # When disabled gates are queried, state() must return False for those gates
    # regardless of the gate's internal tripped flag. So enable spy_ema for this test.
    policy.spy_ema_enabled = True
    assert policy.state(bar_idx=0).book_flat is True
    policy.spy_ema.tripped = False
    policy.vix.tripped = True
    policy.vix_enabled = True
    assert policy.state(bar_idx=0).book_flat is True
    policy.vix.tripped = False
    policy.circuit_breaker.tripped = True
    policy.circuit_breaker_enabled = True
    assert policy.state(bar_idx=0).book_flat is True


def test_disabled_gate_never_trips():
    from backtester.engine.regime import RegimePolicy
    policy = RegimePolicy.from_disabled()
    # All gates disabled -> update is a no-op, tripped stays False.
    aux_data = {
        "SPY": _spy_series([100, 99, 98, 50]),
        "^VIX": _vix_series([20, 40, 50, 60]),
    }
    recent_pnl = pd.Series(
        [-1000] * 4, index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    for i in range(4):
        policy.update(
            bar_idx=i, aux_data=aux_data, recent_pnl=recent_pnl, initial_cash=100_000.0,
        )
    assert policy.state(bar_idx=3).book_flat is False


def test_flatten_on_trip_emits_zero_target_for_all_open():
    from backtester.engine.regime import RegimePolicy
    from backtester.config.models import (
        RegimesConfig, SpyEmaRegimeConfig, VixRegimeConfig, CircuitBreakerConfig,
    )
    cfg = RegimesConfig(
        spy_ema=SpyEmaRegimeConfig(enabled=True, ema_lookback=3, trip_pct=-0.02, resume_pct=0.02),
        vix=VixRegimeConfig(enabled=False),
        circuit_breaker=CircuitBreakerConfig(enabled=False),
    )
    policy = RegimePolicy.from_config(cfg)
    aux_data = {"SPY": _spy_series([100, 100, 100, 70])}  # crash on bar 3
    recent_pnl = pd.Series(
        [0.0] * 4, index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    for i in range(4):
        policy.update(
            bar_idx=i, aux_data=aux_data, recent_pnl=recent_pnl, initial_cash=100_000.0,
        )
    assert policy.state(bar_idx=3).book_flat is True
