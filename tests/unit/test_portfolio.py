from __future__ import annotations

import pandas as pd
import pytest

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from tests.fixtures.synthetic import make_ohlcv


def _buy_and_hold_signals(data: pd.DataFrame) -> SignalFrame:
    sf = pd.DataFrame(index=data.index)
    sf["signal"] = 1
    sf["signal"].iloc[0] = 0  # enter on bar 2 (signal already shifted by strategy convention)
    sf["size"] = 1.0
    return SignalFrame(data=sf)


def test_flat_signal_produces_no_trades(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    flat = SignalFrame(data=pd.DataFrame({"signal": 0, "size": 1.0}, index=ohlcv_small.index))
    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=flat, broker=broker)
    assert len(trades) == 0
    assert eq["equity"].iloc[0] == pytest.approx(10_000.0)
    assert eq["equity"].iloc[-1] == pytest.approx(10_000.0)


def test_signal_change_emits_one_buy_and_one_sell(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                                    initial_cash=10_000.0))
    # signal long for first 30 bars, then flat
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:30] = 1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 2  # one entry, one exit
    assert trades.iloc[0]["side"] == "buy"
    assert trades.iloc[1]["side"] == "sell"
    # equity series has same length as data
    assert len(eq) == n


def test_equity_curve_reflects_pnl():
    data = make_ohlcv(n=50, seed=99, start_price=100.0, drift=0.005, vol=0.001)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    # With positive drift and no costs, equity should rise
    assert eq["equity"].iloc[-1] > eq["equity"].iloc[0]


def test_limit_orders_via_price_column():
    data = make_ohlcv(n=20, seed=11)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = 1
    sig["size"] = 1.0
    # Limit far below market — should not fill on next bar
    sig["limit_price"] = data["low"].min() * 0.5
    sf = SignalFrame(data=sig, price_column="limit_price")
    trades, _, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert len(trades) == 0
