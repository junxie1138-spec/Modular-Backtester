from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.enums import OrderSide
from backtester.engine.fills import Fill
from backtester.engine.position import Position


def _fill(side, qty, price, commission=0.0, ts="2024-01-02"):
    return Fill(timestamp=pd.Timestamp(ts), symbol="SPY",
                side=side, qty=qty, price=price, commission=commission)


def test_position_starts_flat():
    p = Position(symbol="SPY")
    assert p.qty == 0
    assert p.is_flat
    assert p.realized_pnl == 0.0


def test_buy_increases_qty_and_avg_cost():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0, commission=1.0))
    assert p.qty == 10
    assert p.avg_cost == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(-1.0)


def test_two_buys_compute_weighted_avg_cost():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 110.0))
    assert p.qty == 20
    assert p.avg_cost == pytest.approx(105.0)


def test_partial_sell_realizes_pnl():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 4, 110.0, commission=0.5))
    assert p.qty == 6
    assert p.realized_pnl == pytest.approx(4 * (110.0 - 100.0) - 0.5)


def test_full_sell_returns_to_flat():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 10, 105.0))
    assert p.is_flat
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(50.0)


def test_sell_when_flat_raises():
    p = Position(symbol="SPY")
    with pytest.raises(ValueError, match="long-only"):
        p.apply_fill(_fill(OrderSide.SELL, 1, 100.0))


def test_mark_to_market():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    assert p.market_value(price=110.0) == pytest.approx(1100.0)
    assert p.unrealized_pnl(price=110.0) == pytest.approx(100.0)
