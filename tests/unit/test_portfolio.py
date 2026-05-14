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


# --- Short-position simulator tests (Phase 0.2) ---

from backtester.core.enums import OrderSide, OrderType
from backtester.engine.orders import Order
from backtester.engine.position import Position
from backtester.engine.fills import Fill


def _short_broker():
    return Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0, allow_short=True))


def test_flat_to_short_emits_one_sell(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:20] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 2, "expected one short entry + one cover"
    assert trades.iloc[0]["side"] == "sell"
    assert trades.iloc[1]["side"] == "buy"
    # At some point position qty should be negative
    assert (positions["qty"] < 0).any()


def test_short_signal_blocked_when_allow_short_false(ohlcv_small):
    from backtester.core.exceptions import ShortNotAllowedError
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    # Default allow_short=False
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    with pytest.raises(ShortNotAllowedError, match="allow_short"):
        sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)


def test_long_to_short_flip_in_one_order(ohlcv_small):
    """A signal sequence long -> short emits a single SELL that closes the
    long and opens a new short in one fill (combined-order design)."""
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:15] = 1
    # Short region ends well before the last bar so the short->flat
    # transition has an execution bar following it.
    sig["signal"].iloc[15:n - 10] = -1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    # Long entry, flip-to-short combined SELL, cover BUY = 3 fills
    assert len(trades) == 3
    assert list(trades["side"]) == ["buy", "sell", "buy"]
    # The flip SELL qty exceeds the prior long qty (closes long + opens short)
    long_entry_qty = trades.iloc[0]["qty"]
    flip_sell_qty = trades.iloc[1]["qty"]
    assert flip_sell_qty > long_entry_qty
    # Position goes long, then negative
    assert (positions["qty"] > 0).any()
    assert (positions["qty"] < 0).any()


def test_short_to_long_flip_in_one_order(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:15] = -1
    # Long region ends well before the last bar so the long->flat
    # transition has an execution bar following it.
    sig["signal"].iloc[15:n - 10] = 1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 3
    assert list(trades["side"]) == ["sell", "buy", "sell"]
    assert (positions["qty"] < 0).any()
    assert (positions["qty"] > 0).any()


def test_short_entry_via_sell_limit():
    """SELL LIMIT short entry: limit price above current market should fill at
    the limit when the next bar's high reaches it."""
    data = make_ohlcv(n=20, seed=7)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = _short_broker()
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = -1
    sig["size"] = 1.0
    # Place SELL LIMIT slightly above the very high of bar 2 -> should NOT fill
    sig["limit_price"] = data["high"].iloc[2] * 2.0
    sf = SignalFrame(data=sig, price_column="limit_price")
    trades, _, _ = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert len(trades) == 0

    # Now place a SELL LIMIT below the next bar's high -> should fill
    sig2 = sig.copy()
    sig2["limit_price"] = data["low"].iloc[2] * 0.5
    sf2 = SignalFrame(data=sig2, price_column="limit_price")
    trades2, positions2, _ = sim.simulate(data=data, signal_frame=sf2, broker=broker)
    assert len(trades2) >= 1
    assert trades2.iloc[0]["side"] == "sell"
    assert (positions2["qty"] < 0).any()


def test_buy_stop_covers_a_short_when_price_rises_into_stop():
    """STOP support is exercised directly through Broker+Position rather than
    via the simulator's signal->order path (the simulator does not emit STOP
    orders). This verifies the FillEngine + Position wiring for shorts."""
    data = make_ohlcv(n=10, seed=3)
    broker = _short_broker()
    pos = Position(symbol="SPY", allow_short=True)

    # Open a short directly (skip the simulator)
    short_fill = Fill(
        timestamp=data.index[0], symbol="SPY", side=OrderSide.SELL,
        qty=10.0, price=float(data["close"].iloc[0]), commission=0.0,
    )
    pos.apply_fill(short_fill)
    assert pos.qty == -10.0

    # Build a BUY STOP at a level the next bar's high will exceed
    next_bar = data.iloc[1]
    stop_price = float(next_bar["low"])  # guaranteed <= high
    order = Order(
        timestamp=next_bar.name, symbol="SPY",
        side=OrderSide.BUY, qty=10.0,
        order_type=OrderType.STOP, stop_price=stop_price,
    )
    fill = broker.submit(order, next_bar)
    assert fill is not None
    pos.apply_fill(fill)
    assert pos.is_flat
    # cover above short entry means a loss; below means a gain — either way
    # realized_pnl is well-defined and non-NaN
    assert pos.realized_pnl == pos.realized_pnl  # not NaN


def test_long_trailing_stop_fires_on_drawdown():
    # Build a 20-bar series: 10 bars trending up then a sharp drop.
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=20)
    closes = np.concatenate([
        np.linspace(100.0, 120.0, 10),  # uptrend
        np.linspace(118.0, 100.0, 10),  # drawdown ~15%
    ])
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()
    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * 20},
        index=idx,
    )

    # Long-forever signal (enter bar 1, hold).
    sig = pd.DataFrame(index=idx)
    sig["signal"] = 1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    # Exactly one BUY (entry) and one SELL (stop-out).
    assert len(trades) >= 2
    assert trades.iloc[0]["side"] == "buy"
    assert trades.iloc[0]["reason"] == "signal"
    # At least one SELL must be a trailing_stop.
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    assert stop_rows.iloc[0]["side"] == "sell"
    # Position must reach flat after the stop.
    assert (positions["qty"] == 0).any()


def test_short_trailing_stop_fires_on_rally():
    import numpy as np
    idx = pd.bdate_range("2024-01-02", periods=20)
    closes = np.concatenate([
        np.linspace(100.0, 80.0, 10),   # downtrend (good for shorts)
        np.linspace(82.0, 100.0, 10),   # rally (stops out the short)
    ])
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes.copy()
    data = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000] * 20},
        index=idx,
    )

    sig = pd.DataFrame(index=idx)
    sig["signal"] = -1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         allow_short=True, trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert trades.iloc[0]["side"] == "sell"
    assert trades.iloc[0]["reason"] == "signal"
    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) >= 1
    assert stop_rows.iloc[0]["side"] == "buy"
    assert (positions["qty"] < 0).any()
    assert (positions["qty"] == 0).any()


def test_gap_through_stop_fills_at_open():
    """When the bar's open gaps below a long trailing stop, the fill price
    is the bar's open (realistic), not the stop level (charitable)."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    # Bars 1-3: rise gently. Bar 4 gaps DOWN through any 5% stop.
    data = pd.DataFrame({
        "open":   [100.0, 101.0, 103.0, 105.0, 80.0],
        "high":   [101.0, 102.0, 104.0, 106.0, 82.0],
        "low":    [99.5,  100.5, 102.5, 104.5, 78.0],
        "close":  [100.5, 101.5, 103.5, 105.5, 81.0],
        "volume": [1_000_000] * 5,
    }, index=idx)
    # Enter long on bar 1; stop fires on bar 4 (gap-down).
    sig = pd.DataFrame(index=idx)
    sig["signal"] = [0, 1, 1, 1, 1]
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    cfg = ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                         trailing_stop_pct=0.05)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(cfg)
    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)

    stop_rows = trades[trades["reason"] == "trailing_stop"]
    assert len(stop_rows) == 1
    # peak_high before bar 4 is max(101,102,104,106) = 106. stop_level = 106 * 0.95 = 100.7.
    # Bar 4 open = 80, which is BELOW stop_level. Fill price = min(open, stop) = 80.
    assert stop_rows.iloc[0]["price"] == pytest.approx(80.0)
