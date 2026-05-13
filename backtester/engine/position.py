from __future__ import annotations

from dataclasses import dataclass

from backtester.core.enums import OrderSide
from backtester.engine.fills import Fill


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def apply_fill(self, fill: Fill) -> None:
        if fill.side == OrderSide.BUY:
            new_qty = self.qty + fill.qty
            self.avg_cost = (self.avg_cost * self.qty + fill.price * fill.qty) / new_qty
            self.qty = new_qty
            self.realized_pnl -= fill.commission
        else:  # SELL — long-only means we can only close existing longs
            if self.qty <= 0:
                raise ValueError("long-only: cannot SELL when position is flat")
            sell_qty = min(fill.qty, self.qty)
            self.realized_pnl += sell_qty * (fill.price - self.avg_cost) - fill.commission
            self.qty -= sell_qty
            if self.qty == 0:
                self.avg_cost = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_cost)
