from __future__ import annotations

import pandas as pd

from backtester.config.models import ExecutionConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.engine.broker import Broker
from backtester.engine.orders import Order


def test_broker_builds_fill_engine_from_config():
    b = Broker(ExecutionConfig(commission_bps=3.0, slippage_bps=4.0))
    assert b.fills.commission_bps == 3.0
    assert b.fills.slippage_bps == 4.0


def test_broker_submit_returns_fill_for_market():
    b = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    bar = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
                    name=pd.Timestamp("2024-01-02"))
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=5, order_type=OrderType.MARKET)
    fill = b.submit(order, bar)
    assert fill is not None
    assert fill.qty == 5
