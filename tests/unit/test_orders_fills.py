from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.enums import OrderSide, OrderStatus, OrderType
from backtester.engine.orders import Order
from backtester.engine.fills import Fill, FillEngine


def test_market_order_construction():
    o = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)
    assert o.status == OrderStatus.PENDING
    assert o.limit_price is None


def test_limit_order_requires_price():
    with pytest.raises(ValueError, match="limit_price"):
        Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT)


def test_stop_order_requires_price():
    with pytest.raises(ValueError, match="stop_price"):
        Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.STOP)


def _bar(open_, high, low, close):
    return pd.Series({"open": open_, "high": high, "low": low, "close": close, "volume": 1000})


def test_market_buy_fills_at_open_with_slippage():
    fe = FillEngine(commission_bps=1.0, slippage_bps=10.0)
    bar = _bar(100.0, 102.0, 99.0, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(100.0 * (1 + 10e-4))
    assert fill.qty == 10
    expected_cost = fill.price * 10
    assert fill.commission == pytest.approx(expected_cost * 1e-4)


def test_market_sell_fills_at_open_minus_slippage():
    fe = FillEngine(commission_bps=0.0, slippage_bps=20.0)
    bar = _bar(100.0, 102.0, 99.0, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.SELL, qty=5, order_type=OrderType.MARKET)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(100.0 * (1 - 20e-4))


def test_limit_buy_skips_when_low_above_limit():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 102.0, 99.5, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT, limit_price=99.0)
    assert fe.try_fill(order, bar) is None


def test_limit_buy_fills_at_limit_when_touched():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 102.0, 98.5, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT, limit_price=99.0)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(99.0)


def test_stop_buy_skips_when_high_below_stop():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 101.0, 99.5, 100.5)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.STOP, stop_price=102.0)
    assert fe.try_fill(order, bar) is None


def test_stop_buy_fills_at_stop_when_triggered():
    fe = FillEngine(commission_bps=0.0, slippage_bps=10.0)
    bar = _bar(100.0, 103.0, 99.0, 102.5)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.STOP, stop_price=101.0)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    # stop triggered, fills at max(open, stop) + slippage
    assert fill.price == pytest.approx(101.0 * (1 + 10e-4))


def test_fill_dataclass_fields():
    f = Fill(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
             side=OrderSide.BUY, qty=10, price=100.0, commission=1.0)
    assert f.notional == 1000.0
    assert f.cash_delta == -(1000.0 + 1.0)
