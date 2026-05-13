from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd

from backtester.config.models import PortfolioConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position


class PortfolioSimulator:
    """Translates signals -> orders -> fills, tracking cash, position, equity."""

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

        symbol = "ASSET"  # filled in by engine via ctx; for primitives we use a fixed tag
        pos = Position(symbol=symbol)
        cash = self.initial_cash

        fills: List[Fill] = []
        pending: Optional[Order] = None
        prev_signal = 0

        equity_rows = []
        position_rows = []

        index = data.index
        for i, ts in enumerate(index):
            bar = data.iloc[i]
            # 1. Try to execute any pending order on this bar
            if pending is not None:
                fill = broker.submit(pending, bar)
                if fill is not None:
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                pending = None  # one-shot semantics: cancel if not filled

            # 2. Read this bar's signal; if it differs from current state, schedule order for next bar
            sig = int(signals[sig_col].iloc[i]) if sig_col in signals.columns else 0
            if i + 1 < len(index):
                next_bar_ts = index[i + 1]
                target_long = sig == 1
                currently_long = pos.qty > 0

                if target_long and not currently_long:
                    # entry order
                    equity_now = cash + pos.market_value(float(bar["close"]))
                    size = float(signals[size_col].iloc[i]) if size_col and size_col in signals.columns else 1.0
                    alloc = equity_now * self.config.size * size
                    raw_qty = alloc / float(bar["close"])
                    qty = broker.round_qty(raw_qty)
                    if qty > 0:
                        if price_col and price_col in signals.columns and pd.notna(signals[price_col].iloc[i]):
                            pending = Order(
                                timestamp=next_bar_ts, symbol=symbol, side=OrderSide.BUY,
                                qty=qty, order_type=OrderType.LIMIT,
                                limit_price=float(signals[price_col].iloc[i]),
                            )
                        else:
                            pending = Order(
                                timestamp=next_bar_ts, symbol=symbol, side=OrderSide.BUY,
                                qty=qty, order_type=OrderType.MARKET,
                            )

                elif not target_long and currently_long:
                    pending = Order(
                        timestamp=next_bar_ts, symbol=symbol, side=OrderSide.SELL,
                        qty=pos.qty, order_type=OrderType.MARKET,
                    )

            # 3. Mark to market at close
            mv = pos.market_value(float(bar["close"]))
            equity = cash + mv
            equity_rows.append({"timestamp": ts, "cash": cash, "position_value": mv, "equity": equity})
            position_rows.append({"timestamp": ts, "qty": pos.qty, "avg_cost": pos.avg_cost, "close": float(bar["close"])})
            prev_signal = sig

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
