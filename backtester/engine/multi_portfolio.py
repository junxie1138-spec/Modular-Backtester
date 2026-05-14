from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from backtester.core.enums import OrderSide, OrderType
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position


@dataclass
class MultiSymbolResult:
    equity_curve: pd.Series
    final_equity: float
    trades_per_symbol: dict[str, list[Fill]] = field(default_factory=dict)
    portfolio_max_drawdown: float = 0.0
    portfolio_total_return: float = 0.0
    portfolio_sharpe: float = 0.0


@dataclass
class MultiSymbolPortfolioSimulator:
    config: Any  # PortfolioConfig
    initial_cash: float
    broker_factory: Callable[[], Broker]

    def simulate(
        self,
        *,
        symbols: list[str],
        data: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        signals: dict[str, pd.DataFrame],
        aux_data: dict[str, pd.DataFrame],
        regime_config: Optional[Any] = None,
    ) -> MultiSymbolResult:
        """Run the multi-symbol backtest. Skeleton — enforcers wired in later tasks."""
        index = data[symbols[0]].index

        # Per-symbol state.
        brokers: dict[str, Broker] = {s: self.broker_factory() for s in symbols}
        positions: dict[str, Position] = {s: Position(symbol=s) for s in symbols}
        trades: dict[str, list[Fill]] = {s: [] for s in symbols}
        pending_signal: dict[str, Optional[Order]] = {s: None for s in symbols}
        pending_stop: dict[str, Optional[Order]] = {s: None for s in symbols}

        # Shared cash + equity.
        cash = self.initial_cash
        equity_history: list[float] = []

        for i in range(len(index)):
            ts = index[i]

            # Step 1: execute pending stop orders (none in skeleton).
            stop_filled = {s: False for s in symbols}

            # Step 2: execute pending signal orders.
            for s in symbols:
                if pending_signal[s] is not None and not stop_filled[s]:
                    bar = data[s].iloc[i]
                    fill = brokers[s].submit(pending_signal[s], bar=bar)
                    if fill is not None:
                        fill.reason = "signal"
                        cash += fill.cash_delta
                        positions[s].apply_fill(fill)
                        trades[s].append(fill)
                pending_signal[s] = None

            # Step 10: schedule orders for bar i+1.
            if i + 1 < len(index):
                next_ts = index[i + 1]
                portfolio_equity_now = cash + sum(
                    positions[s].qty * float(data[s]["close"].iloc[i]) for s in symbols
                )
                for s in symbols:
                    target = float(signals[s]["signal"].iloc[i])
                    if abs(target) < 1e-12:
                        target_qty = 0
                    else:
                        # percent_equity sizing in skeleton.
                        intent_dollars = target * portfolio_equity_now * self.config.size
                        close_px = float(data[s]["close"].iloc[i])
                        target_qty = int(intent_dollars / close_px) if close_px > 0 else 0
                    delta = target_qty - positions[s].qty
                    if abs(delta) > 1e-9:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        pending_signal[s] = Order(
                            timestamp=next_ts,
                            symbol=s,
                            side=side,
                            qty=abs(delta),
                            order_type=OrderType.MARKET,
                        )

            # Step 11: mark to market.
            position_value = sum(
                positions[s].qty * float(data[s]["close"].iloc[i]) for s in symbols
            )
            equity_history.append(cash + position_value)

        equity_curve = pd.Series(equity_history, index=index, name="equity")
        return MultiSymbolResult(
            equity_curve=equity_curve,
            final_equity=float(equity_curve.iloc[-1]),
            trades_per_symbol=trades,
            portfolio_total_return=float(equity_curve.iloc[-1]) / self.initial_cash - 1.0,
        )
