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
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY")  # allow_short defaults to False
    with pytest.raises(ShortNotAllowedError, match="shorts not allowed"):
        p.apply_fill(_fill(OrderSide.SELL, 1, 100.0))


def test_mark_to_market():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    assert p.market_value(price=110.0) == pytest.approx(1100.0)
    assert p.unrealized_pnl(price=110.0) == pytest.approx(100.0)


# --- Short-side tests (Phase 0.2 short-position support) ---

def _short_pos():
    """Helper: position with shorts enabled."""
    return Position(symbol="SPY", allow_short=True)


def test_short_entry_from_flat_sets_negative_qty_and_avg_cost():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0, commission=1.0))
    assert p.qty == -10
    assert p.avg_cost == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(-1.0)
    assert not p.is_flat


def test_two_short_entries_compute_weighted_avg_cost():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 10, 110.0))
    assert p.qty == -20
    assert p.avg_cost == pytest.approx(105.0)


def test_partial_cover_realizes_short_pnl_when_price_drops():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))            # short @ 100
    p.apply_fill(_fill(OrderSide.BUY, 4, 90.0, commission=0.5))  # cover 4 @ 90
    assert p.qty == -6
    # Short PnL: sell_qty * (avg_cost - cover_price) - commission
    assert p.realized_pnl == pytest.approx(4 * (100.0 - 90.0) - 0.5)
    # avg_cost unchanged on partial close
    assert p.avg_cost == pytest.approx(100.0)


def test_full_cover_returns_to_flat():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 95.0))
    assert p.is_flat
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(50.0)  # 10 * (100 - 95)


def test_losing_short_realizes_negative_pnl():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 110.0))
    assert p.is_flat
    assert p.realized_pnl == pytest.approx(-100.0)  # 10 * (100 - 110)


def test_short_mark_to_market_negative_value():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))
    # market_value(qty * price) is negative when short
    assert p.market_value(price=95.0) == pytest.approx(-950.0)
    # unrealized PnL = qty * (price - avg_cost) = -10 * (95 - 100) = +50
    assert p.unrealized_pnl(price=95.0) == pytest.approx(50.0)


def test_long_to_short_flip_in_one_fill():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))   # long 10 @ 100
    p.apply_fill(_fill(OrderSide.SELL, 15, 105.0))  # sell 15 -> flat then short 5
    assert p.qty == -5
    assert p.avg_cost == pytest.approx(105.0)
    # realized = closed-long PnL only: 10 * (105 - 100) = 50
    assert p.realized_pnl == pytest.approx(50.0)


def test_short_to_long_flip_in_one_fill():
    p = _short_pos()
    p.apply_fill(_fill(OrderSide.SELL, 10, 100.0))  # short 10 @ 100
    p.apply_fill(_fill(OrderSide.BUY, 15, 95.0))    # buy 15 -> flat then long 5
    assert p.qty == 5
    assert p.avg_cost == pytest.approx(95.0)
    # realized = closed-short PnL only: 10 * (100 - 95) = 50
    assert p.realized_pnl == pytest.approx(50.0)


def test_long_to_short_flip_blocked_when_allow_short_false():
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY")  # allow_short=False
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    with pytest.raises(ShortNotAllowedError):
        p.apply_fill(_fill(OrderSide.SELL, 15, 105.0))


def test_short_open_blocked_when_allow_short_false():
    """Same intent as test_sell_when_flat_raises but exercised by an
    explicit allow_short=False construction. Belt-and-suspenders."""
    from backtester.core.exceptions import ShortNotAllowedError
    p = Position(symbol="SPY", allow_short=False)
    with pytest.raises(ShortNotAllowedError):
        p.apply_fill(_fill(OrderSide.SELL, 5, 100.0))
