from __future__ import annotations

from dataclasses import dataclass

from backtester.core.enums import OrderSide
from backtester.core.exceptions import ShortNotAllowedError
from backtester.engine.fills import Fill


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    allow_short: bool = False

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def apply_fill(self, fill: Fill) -> None:
        signed_delta = fill.qty if fill.side == OrderSide.BUY else -fill.qty
        original_qty = self.qty
        new_qty = original_qty + signed_delta

        same_sign_or_flat = (
            original_qty == 0
            or (original_qty > 0 and signed_delta > 0)
            or (original_qty < 0 and signed_delta < 0)
        )

        if same_sign_or_flat:
            # Opening from flat, or growing existing direction.
            if signed_delta < 0 and original_qty == 0 and not self.allow_short:
                raise ShortNotAllowedError(
                    "cannot SELL when position is flat: shorts not allowed"
                )
            if original_qty == 0:
                self.avg_cost = fill.price
            else:
                total_abs = abs(original_qty) + abs(signed_delta)
                self.avg_cost = (
                    self.avg_cost * abs(original_qty)
                    + fill.price * abs(signed_delta)
                ) / total_abs
            self.qty = new_qty
        else:
            # Opposite sign: realize PnL on the closed portion, possibly flip.
            close_qty = min(abs(signed_delta), abs(original_qty))
            sign = 1 if original_qty > 0 else -1
            # TODO(short-positions): borrow cost / hard-to-borrow modeling is
            # not included here. A future phase should accrue daily borrow fee
            # against realized_pnl while qty < 0.
            self.realized_pnl += sign * close_qty * (fill.price - self.avg_cost)
            self.qty = new_qty
            if new_qty == 0:
                self.avg_cost = 0.0
            elif (new_qty > 0) != (original_qty > 0):
                # Flipped through zero — leftover opens a fresh position.
                if new_qty < 0 and not self.allow_short:
                    raise ShortNotAllowedError(
                        "cannot flip long->short: shorts not allowed"
                    )
                self.avg_cost = fill.price
            # else: partial close in same direction — avg_cost unchanged.

        self.realized_pnl -= fill.commission

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_cost)
