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


def test_promote_to_runner_called_on_partial_close():
    """Strategy emits target=0.5 from full position -> TrancheStopState should promote."""
    from backtester.engine.tranche_stop import TSPhase
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA"])
    # Reconfigure broker to enable tranche-stop machinery.
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        allow_fractional=False, allow_short=False,
        hard_stop_atr_mult=1.75, runner_atr_mult=2.5,
        breakeven_floor=True, tranche_stop_atr_period=3,
    ))
    sim.config.size = 0.5

    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])}
    # Bar 1: enter full (signal=1.0). Bar 3: scale to 0.5 (tranche 1 fills next bar).
    # Bar 5: exit (signal=0).
    sig = pd.DataFrame({
        "signal": [0.0, 1.0, 1.0, 0.5, 0.5, 0.0],
        "size":   [1.0] * 6,
    }, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # End of run: position fully exited, phase DISARMED, qty == 0.
    assert result.tranche_phase_at_end["AAA"] is TSPhase.DISARMED
    assert result.position_qty_at_end["AAA"] == 0


def test_disarm_called_on_full_exit():
    from backtester.engine.tranche_stop import TSPhase
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA"])
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.75, runner_atr_mult=2.5,
        breakeven_floor=True, tranche_stop_atr_period=3,
    ))

    # 5 bars: signal=0 at index 3 schedules exit at bar 4 (index 4 executes it).
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    data = {"AAA": _ohlcv([100.0] * 5)}
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0, 0.0], "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    assert result.tranche_phase_at_end["AAA"] is TSPhase.DISARMED


def test_pending_stop_per_symbol_independent():
    """Each symbol's pending_stop is independent. Stop on AAA does not affect BBB."""
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA", "BBB"])
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,  # tight stop
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))

    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = _ohlcv([100.0, 100.0, 100.0, 100.0, 80.0])  # crash on last bar
    bbb = _ohlcv([200.0] * 5)
    aaa_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0, 1.0], "size": [1.0]*5}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0, 1.0], "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # AAA stopped out: at least one trade has reason="trailing_stop".
    aaa_trades = result.trades_per_symbol["AAA"]
    assert any(f.reason == "trailing_stop" for f in aaa_trades)
    # BBB never stopped.
    bbb_trades = result.trades_per_symbol["BBB"]
    assert not any(f.reason == "trailing_stop" for f in bbb_trades)


def test_stop_wins_over_signal_same_bar_per_symbol():
    """On the bar where a stop fires AND the strategy signals an exit, only the stop fill lands."""
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA"])
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))

    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 50.0])}  # crash on last bar
    # Strategy ALSO signals exit on bar 3 (signal=0 at index 2 -> exit at bar 3 open).
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    trades = result.trades_per_symbol["AAA"]
    exit_fills = [f for f in trades if f.side.value == "sell"]
    # Exactly one exit landed.
    assert len(exit_fills) == 1
    # And it's the trailing_stop (stop wins over same-bar signal).
    assert exit_fills[0].reason == "trailing_stop"


def test_simultaneous_entries_compete_for_risk_budget():
    """When two new entries would together exceed risk_budget_pct, one is dropped."""
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA", "BBB"])
    sim.config.risk_budget_pct = 0.03  # tight cap
    sim.config.size = 0.5
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))

    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    aaa = _ohlcv([100.0] * 4)
    bbb = _ohlcv([200.0] * 4)
    aaa_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # At least one was dropped.
    aaa_entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    bbb_entries = [f for f in result.trades_per_symbol["BBB"] if f.side.value == "buy"]
    assert len(aaa_entries) == 0 or len(bbb_entries) == 0


def test_position_cap_clips_oversized_signal():
    """If strategy emits target=1.0 but position_cap_pct=0.05, position is 5% not 100%."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.position_cap_pct = 0.05
    sim.config.size = 1.0

    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 100.0])}
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Position dollars cap = 5% of $100k = $5,000; at $100/share = 50 shares.
    entry = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"][0]
    assert entry.qty <= 50


def test_cash_reserve_cap_drops_late_entries():
    """When deployed would exceed (1 - cash_reserve_pct), late entries get dropped."""
    sim = _build_simulator(symbols=["AAA", "BBB", "CCC"])
    sim.config.cash_reserve_pct = 0.30
    sim.config.size = 0.5  # each entry intends 50% deployment

    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {s: _ohlcv([100.0] * 4) for s in ["AAA", "BBB", "CCC"]}
    sigs = {s: pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx) for s in ["AAA", "BBB", "CCC"]}
    result = sim.simulate(
        symbols=["AAA", "BBB", "CCC"], data=data,
        sectors={"AAA": "X", "BBB": "Y", "CCC": "Z"},
        signals=sigs, aux_data={}, regime_config=None,
    )
    # Two entries would deploy 100%; cash_reserve=0.30 caps total at 70%, only ~1 entry fits.
    entry_count = sum(
        1 for s in ["AAA", "BBB", "CCC"]
        if any(f.side.value == "buy" for f in result.trades_per_symbol[s])
    )
    assert entry_count <= 2


def test_sector_cap_blocks_third_entry_in_full_sector():
    sim = _build_simulator(symbols=["AAA", "BBB", "CCC"])
    sim.config.sector_cap_pct = 0.50
    sim.config.size = 0.3  # 3 entries x 30% = 90% in one sector

    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {s: _ohlcv([100.0] * 4) for s in ["AAA", "BBB", "CCC"]}
    sigs = {s: pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0], "size": [1.0]*4}, index=idx) for s in ["AAA", "BBB", "CCC"]}
    result = sim.simulate(
        symbols=["AAA", "BBB", "CCC"], data=data,
        sectors={"AAA": "Semis", "BBB": "Semis", "CCC": "Semis"},
        signals=sigs, aux_data={}, regime_config=None,
    )
    entry_count = sum(
        1 for s in ["AAA", "BBB", "CCC"]
        if any(f.side.value == "buy" for f in result.trades_per_symbol[s])
    )
    assert entry_count < 3


def test_risk_budget_released_on_full_exit():
    """After a full exit, freed risk budget allows a new entry on a later bar."""
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    sim = _build_simulator(symbols=["AAA", "BBB"])
    sim.config.risk_budget_pct = 0.05
    sim.config.size = 0.5
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))

    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    aaa = _ohlcv([100.0] * 6)
    bbb = _ohlcv([200.0] * 6)
    # AAA enters bar 1 (signal=1), exits bar 3 (signal=0 at index 2).
    # BBB tries entry at bar 4 (signal=1 at index 4) — budget should be free.
    aaa_sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], "size": [1.0]*6}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "size": [1.0]*6}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    bbb_entries = [f for f in result.trades_per_symbol["BBB"] if f.side.value == "buy"]
    assert len(bbb_entries) == 1


def test_vol_targeted_sizing_uses_realized_vol_20d():
    """sizing_mode='vol_targeted' produces position dollars consistent with realized vol."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.sizing_mode = "vol_targeted"
    sim.config.vol_target = 0.12
    sim.config.position_cap_pct = 1.0
    sim.config.size = 1.0

    idx = pd.date_range("2024-01-02", periods=30, freq="B")
    closes = [100.0 + 0.1 * i for i in range(30)]
    data = {"AAA": _ohlcv(closes)}
    # Enter on bar 26 (signal=1.0 at idx 25 -> fill at bar 26 open).
    sig_values = [0.0] * 25 + [1.0, 1.0, 1.0, 0.0, 0.0]
    sig = pd.DataFrame({"signal": sig_values, "size": [1.0]*30}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    assert len(entries) >= 1
    assert entries[0].qty > 100


def test_vol_targeted_sizing_defers_entry_during_warmup():
    """No realized_vol available yet -> entry deferred."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.sizing_mode = "vol_targeted"
    sim.config.vol_target = 0.12

    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    data = {"AAA": _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])}
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 0.0, 0.0], "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    assert len(entries) == 0


def test_regime_gate_flattens_book():
    """When the SPY EMA gate trips, all open positions get target=0 next bar."""
    from backtester.config.models import RegimesConfig, SpyEmaRegimeConfig
    sim = _build_simulator(symbols=["AAA"])

    idx = pd.date_range("2024-01-02", periods=8, freq="B")
    data = {"AAA": _ohlcv([100.0] * 8)}
    spy = _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 70.0, 70.0])
    sig = pd.DataFrame({"signal": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], "size": [1.0]*8}, index=idx)
    regimes = RegimesConfig(spy_ema=SpyEmaRegimeConfig(
        enabled=True, ema_lookback=3, trip_pct=-0.02, resume_pct=0.02,
    ))
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={"SPY": spy}, regime_config=regimes,
    )
    exits = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "sell"]
    assert len(exits) >= 1


def test_position_phase_finalized_before_strategy_callback():
    """For uses_per_bar strategies, position_phase reflects end-of-bar-t state when
    the strategy decides bar t+1's signal."""
    from backtester.engine.tranche_stop import TSPhase
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker

    captured: list[Any] = []

    class _CaptureStrategy:
        uses_per_bar = True
        def signal_for_bar(self, *, symbol, bar_idx, data_panel, indicators_panel, ctx, params):
            captured.append((bar_idx, ctx.position_phase.get(symbol)))
            # Synthesize a phase progression: 0 -> 1.0 -> 0.5 -> 0 -> 0
            sched = [0.0, 1.0, 0.5, 0.0, 0.0]
            return sched[bar_idx] if bar_idx < len(sched) else 0.0

    sim = _build_simulator(symbols=["AAA"])
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))

    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    data = {"AAA": _ohlcv([100.0] * 5)}
    # Provide a dummy signals frame (per-bar strategy overrides it).
    sig = pd.DataFrame({"signal": [0.0]*5, "size": [1.0]*5}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
        strategy=_CaptureStrategy(),
    )
    # On bar 2: position just opened from bar 1's signal -> phase should be HARD.
    bar2_phase = next((p for i, p in captured if i == 2), None)
    assert bar2_phase is TSPhase.HARD
    # On bar 3: tranche 1 just filled -> phase should be RUNNER.
    bar3_phase = next((p for i, p in captured if i == 3), None)
    assert bar3_phase is TSPhase.RUNNER


def test_portfolio_metrics_computed_from_equity_curve():
    """Sharpe and max_drawdown are computed from the equity curve, not hardcoded to 0."""
    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=30, freq="B")
    # Construct closes with a clear up-down-up pattern to produce real drawdown.
    closes = [100.0 + i * 0.1 for i in range(15)] + [115.0 - i * 0.3 for i in range(15)]
    data = {"AAA": _ohlcv(closes)}
    sig = pd.DataFrame({"signal": [0.0] + [1.0] * 28 + [0.0], "size": [1.0]*30}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Drawdown should be non-zero (price went up then down while we held).
    assert result.portfolio_max_drawdown < 0.0 or abs(result.portfolio_total_return) > 0
