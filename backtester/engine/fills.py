from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from backtester.core.constants import BPS
from backtester.core.enums import OrderSide, OrderStatus, OrderType
from backtester.engine.orders import Order


@dataclass(slots=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def cash_delta(self) -> float:
        sign = -1.0 if self.side == OrderSide.BUY else 1.0
        return sign * self.notional - self.commission


class FillEngine:
    """Apply commission + slippage and decide whether each order fills on a bar."""

    def __init__(self, commission_bps: float, slippage_bps: float):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps * BPS

    def _slip(self, price: float, side: OrderSide) -> float:
        adj = 1.0 + self.slippage_bps * BPS if side == OrderSide.BUY else 1.0 - self.slippage_bps * BPS
        return price * adj

    def try_fill(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        price: Optional[float] = None

        if order.order_type == OrderType.MARKET:
            price = self._slip(open_, order.side)

        elif order.order_type == OrderType.LIMIT:
            lp = float(order.limit_price)  # type: ignore[arg-type]
            if order.side == OrderSide.BUY and low <= lp:
                price = min(lp, open_)
            elif order.side == OrderSide.SELL and high >= lp:
                price = max(lp, open_)

        elif order.order_type == OrderType.STOP:
            sp = float(order.stop_price)  # type: ignore[arg-type]
            if order.side == OrderSide.BUY and high >= sp:
                triggered = max(open_, sp)
                price = self._slip(triggered, order.side)
            elif order.side == OrderSide.SELL and low <= sp:
                triggered = min(open_, sp)
                price = self._slip(triggered, order.side)

        if price is None:
            return None

        notional = price * order.qty
        commission = self._commission(notional)
        order.status = OrderStatus.FILLED
        return Fill(
            timestamp=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(bar.name),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            commission=commission,
        )
