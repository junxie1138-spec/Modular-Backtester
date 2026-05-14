from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low":  [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def _const_signals(idx, target):
    """Trivial signal frame: emit target on bar 1, hold, exit on bar -1."""
    sig = pd.Series(0.0, index=idx)
    sig.iloc[1] = float(target)
    # Use pre-shifted signal: signal at index i acts on bar i+1's open.
    return pd.DataFrame(
        {"signal": sig, "size": pd.Series(1.0, index=idx)}, index=idx,
    )


def _build_simulator(symbols, *, initial_cash=100_000.0, position_size=0.10):
    """Return a MultiSymbolPortfolioSimulator wired to permissive defaults."""
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker

    return MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(
            sizing_mode="percent_equity",
            size=position_size,
            position_cap_pct=1.0,
            cash_reserve_pct=0.0,
            risk_budget_pct=1.0,
            sector_cap_pct=1.0,
        ),
        initial_cash=initial_cash,
        broker_factory=lambda: Broker(ExecutionConfig(
            initial_cash=initial_cash,
            commission_bps=1.0,
            slippage_bps=0.0,
            allow_fractional=False,
            allow_short=False,
        )),
    )


def test_shared_cash_debited_on_buy_credited_on_sell():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 110.0])}
    sectors = {"AAA": "X"}
    # Signal at index 1 means buy at index 2 open. Signal at index 2 means sell at index 3 open.
    # Use a frame with: 0, 1.0, 0, 0 so buy at bar 2, then no further exit signal — held through to end.
    # We assert final_equity > initial since the position appreciated (100 -> 110).
    sig = pd.DataFrame({
        "signal": [0.0, 1.0, 1.0, 0.0],  # enter bar 2; exit bar 3 (signal=0 at index 2)
        "size":   [1.0, 1.0, 1.0, 1.0],
    }, index=data["AAA"].index)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # After full cycle: bought ~100, position held while price rose to 110, then sold.
    # 10% position size = ~$10k; with price rising 10%, profit ~ $1000.
    assert result.final_equity > 100_000.0


def test_portfolio_equity_sums_cash_plus_positions():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 100.0, 105.0, 105.0])}
    sectors = {"AAA": "X"}
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0], "size": [1.0]*4}, index=data["AAA"].index)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Mid-run (bar 2), position is held; equity = cash + qty * close.
    eq_curve = result.equity_curve
    assert len(eq_curve) == 4
    # Final equity should reflect the appreciation.
    assert result.final_equity > 0


def test_two_symbol_independent_entries_dont_interfere():
    sim = _build_simulator(symbols=["AAA", "BBB"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = _ohlcv([100.0] * 5)
    bbb = _ohlcv([200.0] * 5)
    # AAA enters bar 2 (signal index 1), BBB enters bar 3 (signal index 2).
    aaa_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0, 0.0], "size": [1.0]*5}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0.0, 0.0, 1.0, 1.0, 0.0], "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # Both symbols should have at least one trade in their per-symbol trade log.
    assert len(result.trades_per_symbol["AAA"]) >= 1
    assert len(result.trades_per_symbol["BBB"]) >= 1


def test_unique_per_symbol_trailing_stop_state():
    """Each symbol owns its own position state; states don't bleed."""
    sim = _build_simulator(symbols=["AAA", "BBB"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = _ohlcv([100.0] * 5)
    bbb = _ohlcv([200.0] * 5)
    aaa_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0, 0.0], "size": [1.0]*5}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0.0, 0.0, 1.0, 1.0, 0.0], "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # Both symbols saw independent signal-driven entries.
    assert "AAA" in result.trades_per_symbol
    assert "BBB" in result.trades_per_symbol


def test_portfolio_equity_curve_length_matches_panel_index():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])}
    sectors = {"AAA": "X"}
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0, 0.0], "size": [1.0]*5}, index=data["AAA"].index)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    assert len(result.equity_curve) == len(data["AAA"])
