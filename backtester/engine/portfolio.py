from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd

from backtester.config.models import PortfolioConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.core.exceptions import ShortNotAllowedError
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position


def _sign(qty: float) -> int:
    if qty > 0:
        return 1
    if qty < 0:
        return -1
    return 0


class PortfolioSimulator:
    """Translates signals -> orders -> fills, tracking cash, position, equity.

    Signal convention: signals in {-1, 0, 1}. A signal == -1 is rejected
    unless broker.allow_short is True. State transitions are computed from
    (sign(pos.qty), signal); same-sign cases emit no order (no rebalance).
    long<->short flips emit a single combined order; Position.apply_fill
    handles the close + reopen in one fill.
    """

    def __init__(self, config: PortfolioConfig, initial_cash: float = 100_000.0):
        self.config = config
        self.initial_cash = initial_cash

    def simulate(
        self,
        data: pd.DataFrame,
        signal_frame: SignalFrame,
        broker: Broker,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        signals = signal_frame.data
        sig_col = signal_frame.signal_column
        size_col = signal_frame.size_column
        price_col = signal_frame.price_column

        symbol = "ASSET"
        pos = Position(symbol=symbol, allow_short=broker.allow_short)
        cash = self.initial_cash

        fills: List[Fill] = []
        pending: Optional[Order] = None

        equity_rows = []
        position_rows = []

        index = data.index
        for i, ts in enumerate(index):
            bar = data.iloc[i]

            # 1. Execute pending order
            if pending is not None:
                fill = broker.submit(pending, bar)
                if fill is not None:
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                pending = None  # one-shot semantics

            # 2. Read this bar's signal
            sig = int(signals[sig_col].iloc[i]) if sig_col in signals.columns else 0
            if sig == -1 and not broker.allow_short:
                raise ShortNotAllowedError(
                    f"strategy emitted SHORT signal at bar {i} ({ts}) but "
                    f"execution.allow_short is False"
                )

            # 3. Decide whether to schedule an order for the next bar
            if i + 1 < len(index):
                next_bar_ts = index[i + 1]
                prev_sign = _sign(pos.qty)
                target_sign = sig

                if prev_sign != target_sign:
                    close_px = float(bar["close"])
                    if target_sign == 0:
                        # Close current position fully.
                        order_qty = abs(pos.qty)
                        side = OrderSide.BUY if prev_sign < 0 else OrderSide.SELL
                        order_type = OrderType.MARKET
                        limit_price = None
                    else:
                        equity_now = cash + pos.market_value(close_px)
                        size = (
                            float(signals[size_col].iloc[i])
                            if size_col and size_col in signals.columns
                            else 1.0
                        )
                        alloc = equity_now * self.config.size * size
                        new_leg_qty = broker.round_qty(alloc / close_px)
                        if prev_sign == 0:
                            order_qty = new_leg_qty
                        else:
                            # Flip: close old leg + open new leg in one fill.
                            order_qty = abs(pos.qty) + new_leg_qty
                        side = OrderSide.BUY if target_sign > 0 else OrderSide.SELL
                        # LIMIT only when entering from flat.
                        if (
                            prev_sign == 0
                            and price_col
                            and price_col in signals.columns
                            and pd.notna(signals[price_col].iloc[i])
                        ):
                            order_type = OrderType.LIMIT
                            limit_price = float(signals[price_col].iloc[i])
                        else:
                            order_type = OrderType.MARKET
                            limit_price = None

                    if order_qty > 0:
                        pending = Order(
                            timestamp=next_bar_ts,
                            symbol=symbol,
                            side=side,
                            qty=order_qty,
                            order_type=order_type,
                            limit_price=limit_price,
                        )

            # 4. Mark to market at close
            mv = pos.market_value(float(bar["close"]))
            equity = cash + mv
            equity_rows.append({"timestamp": ts, "cash": cash, "position_value": mv, "equity": equity})
            position_rows.append({"timestamp": ts, "qty": pos.qty, "avg_cost": pos.avg_cost, "close": float(bar["close"])})

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        positions_df = pd.DataFrame(position_rows).set_index("timestamp")
        trades_df = pd.DataFrame([
            {
                "timestamp": f.timestamp,
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "commission": f.commission,
                "notional": f.notional,
            }
            for f in fills
        ])
        return trades_df, positions_df, equity_curve
