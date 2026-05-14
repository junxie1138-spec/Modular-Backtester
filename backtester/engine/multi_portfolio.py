from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from backtester.core.enums import OrderSide, OrderType
from backtester.engine.atr import compute_atr
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position
from backtester.engine.tranche_stop import TrancheStopState, TSPhase


@dataclass
class MultiSymbolResult:
    equity_curve: pd.Series
    final_equity: float
    trades_per_symbol: dict[str, list[Fill]] = field(default_factory=dict)
    portfolio_max_drawdown: float = 0.0
    portfolio_total_return: float = 0.0
    portfolio_sharpe: float = 0.0
    tranche_phase_at_end: dict[str, Any] = field(default_factory=dict)
    position_qty_at_end: dict[str, float] = field(default_factory=dict)


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
        """Run the multi-symbol backtest."""
        index = data[symbols[0]].index

        # Per-symbol state.
        brokers: dict[str, Broker] = {s: self.broker_factory() for s in symbols}
        positions: dict[str, Position] = {s: Position(symbol=s) for s in symbols}
        trades: dict[str, list[Fill]] = {s: [] for s in symbols}
        pending_signal: dict[str, Optional[Order]] = {s: None for s in symbols}
        pending_stop: dict[str, Optional[Order]] = {s: None for s in symbols}

        # Build per-symbol TrancheStopState if ExecutionConfig has v0.4.0 keys.
        ts_states: dict[str, TrancheStopState] = {}
        sample_broker = next(iter(brokers.values()))
        ex = sample_broker.config
        if ex.hard_stop_atr_mult is not None:
            for s in symbols:
                ts_states[s] = TrancheStopState(
                    hard_stop_atr_mult=ex.hard_stop_atr_mult,
                    runner_atr_mult=ex.runner_atr_mult,
                    breakeven_floor=ex.breakeven_floor,
                    atr_series=compute_atr(data[s], ex.tranche_stop_atr_period),
                )

        # Shared cash + equity.
        cash = self.initial_cash
        equity_history: list[float] = []

        for i in range(len(index)):
            ts = index[i]

            # Snapshot qty before any fills on this bar.
            prev_qty = {s: positions[s].qty for s in symbols}

            # Step 1: execute pending stop orders.
            stop_filled = {s: False for s in symbols}
            for s in symbols:
                if pending_stop[s] is not None:
                    bar = data[s].iloc[i]
                    fill = brokers[s].submit(pending_stop[s], bar=bar)
                    if fill is not None:
                        fill.reason = "trailing_stop"
                        cash += fill.cash_delta
                        positions[s].apply_fill(fill)
                        trades[s].append(fill)
                        stop_filled[s] = True
                    pending_stop[s] = None

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

            # Step 4: tranche-state transitions per symbol.
            for s in symbols:
                if s not in ts_states:
                    continue
                prev = prev_qty[s]
                new = positions[s].qty
                if prev == 0 and new != 0:
                    # Flat -> non-flat: reset.
                    last_fill = trades[s][-1]
                    ts_states[s].reset(entry_price=last_fill.price, bar_idx=i)
                elif prev != 0 and new == 0:
                    # Non-flat -> flat: disarm (whether stop- or signal-driven).
                    ts_states[s].disarm()
                elif prev != 0 and new != 0 and (prev > 0) == (new > 0) and abs(new) < abs(prev):
                    # Same-sign partial close -> promote.
                    ts_states[s].promote_to_runner()

            # Step 5: update peak/trough on close.
            for s in symbols:
                if s in ts_states:
                    ts_states[s].update(data[s].iloc[i])

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

                    # Schedule stop order for next bar.
                    if (
                        s in ts_states
                        and ts_states[s].phase is not TSPhase.DISARMED
                        and positions[s].qty != 0
                    ):
                        sign = 1 if positions[s].qty > 0 else -1
                        stop_px = ts_states[s].stop_price(sign=sign, bar_idx=i + 1)
                        if stop_px is not None:
                            stop_side = OrderSide.SELL if sign > 0 else OrderSide.BUY
                            pending_stop[s] = Order(
                                symbol=s,
                                side=stop_side,
                                qty=abs(positions[s].qty),
                                order_type=OrderType.STOP,
                                stop_price=stop_px,
                                timestamp=next_ts,
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
            tranche_phase_at_end={
                s: ts_states[s].phase if s in ts_states else None for s in symbols
            },
            position_qty_at_end={s: positions[s].qty for s in symbols},
        )
