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
from backtester.engine.risk_budget import RiskBudgetEnforcer
from backtester.engine.sector_cap import SectorCapEnforcer
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
                portfolio_equity_now = cash + sum(
                    positions[s].qty * float(data[s]["close"].iloc[i]) for s in symbols
                )
                deployed_total = sum(
                    abs(positions[s].qty) * float(data[s]["close"].iloc[i]) for s in symbols
                )
                deployed_per_sector: dict[str, float] = {}
                for s in symbols:
                    sec = sectors[s]
                    deployed_per_sector[sec] = deployed_per_sector.get(sec, 0.0) + (
                        abs(positions[s].qty) * float(data[s]["close"].iloc[i])
                    )
                current_risk_dollars = 0.0
                for s in symbols:
                    if s in ts_states and ts_states[s].phase is not TSPhase.DISARMED:
                        sgn = 1 if positions[s].qty > 0 else -1 if positions[s].qty < 0 else 0
                        if sgn != 0:
                            stop_px = ts_states[s].stop_price(sign=sgn, bar_idx=i)
                            if stop_px is not None:
                                current_risk_dollars += abs(positions[s].qty) * abs(
                                    float(data[s]["close"].iloc[i]) - stop_px
                                )

                risk_enforcer = RiskBudgetEnforcer(budget_pct=self.config.risk_budget_pct)
                sector_enforcer = SectorCapEnforcer(cap_pct=self.config.sector_cap_pct)
                cash_reserve_limit = portfolio_equity_now * (1.0 - self.config.cash_reserve_pct)

                next_ts = index[i + 1]
                # Mutable totals updated as each symbol is approved, so later symbols see
                # the cumulative deployment of earlier symbols in the same scheduling pass.
                running_deployed_total = deployed_total
                running_deployed_per_sector = dict(deployed_per_sector)
                running_risk_dollars = current_risk_dollars
                for s in symbols:
                    target = float(signals[s]["signal"].iloc[i])
                    capped = max(-1.0, min(1.0, target))
                    close_px = float(data[s]["close"].iloc[i])
                    # Apply position_cap_pct.
                    intent_dollars = capped * portfolio_equity_now * self.config.size
                    cap_dollars = portfolio_equity_now * self.config.position_cap_pct
                    if intent_dollars > cap_dollars:
                        intent_dollars = cap_dollars
                    elif intent_dollars < -cap_dollars:
                        intent_dollars = -cap_dollars

                    existing_dollars = abs(positions[s].qty) * close_px
                    proposed_dollars = abs(intent_dollars)

                    # Apply cash reserve, sector cap, and risk budget ONLY for additional deployment.
                    if proposed_dollars > existing_dollars:
                        additional = proposed_dollars - existing_dollars
                        # Cash reserve: block if new total deployment would meet or exceed limit.
                        if running_deployed_total + additional >= cash_reserve_limit:
                            intent_dollars = existing_dollars * (1 if intent_dollars > 0 else -1)
                            proposed_dollars = existing_dollars
                        else:
                            # Sector cap.
                            sec_decision = sector_enforcer.evaluate(
                                sector=sectors[s], deployed_per_sector=running_deployed_per_sector,
                                deployed_total=running_deployed_total, proposed_dollars=additional,
                            )
                            if not sec_decision.admitted:
                                intent_dollars = existing_dollars * (1 if intent_dollars > 0 else -1)
                                proposed_dollars = existing_dollars
                            else:
                                # Risk budget (uses ATR-based stop distance estimate or
                                # a position-fraction proxy when ts_states is unavailable).
                                if s in ts_states:
                                    atr_now = compute_atr(data[s], ex.tranche_stop_atr_period).iloc[i]
                                    if not pd.isna(atr_now) and float(atr_now) > 0:
                                        est_stop_dist = ex.hard_stop_atr_mult * float(atr_now)
                                        if close_px > 0:
                                            est_shares = additional / close_px
                                            proposed_risk = est_shares * est_stop_dist
                                            risk_decision = risk_enforcer.evaluate(
                                                portfolio_equity=portfolio_equity_now,
                                                current_risk_dollars=running_risk_dollars,
                                                proposed_risk_dollars=proposed_risk,
                                            )
                                            if not risk_decision.admitted:
                                                intent_dollars = existing_dollars * (1 if intent_dollars > 0 else -1)
                                                proposed_dollars = existing_dollars
                                            else:
                                                running_risk_dollars += proposed_risk
                                    else:
                                        # ATR is zero/NaN — use a flat 1% of position as proxy risk.
                                        if close_px > 0:
                                            est_shares = additional / close_px
                                            proposed_risk = est_shares * close_px * 0.01
                                            risk_decision = risk_enforcer.evaluate(
                                                portfolio_equity=portfolio_equity_now,
                                                current_risk_dollars=running_risk_dollars,
                                                proposed_risk_dollars=proposed_risk,
                                            )
                                            if not risk_decision.admitted:
                                                intent_dollars = existing_dollars * (1 if intent_dollars > 0 else -1)
                                                proposed_dollars = existing_dollars
                                            else:
                                                running_risk_dollars += proposed_risk
                                else:
                                    # No ts_states: use position dollars * 1% as proxy risk.
                                    proposed_risk = additional * 0.01
                                    risk_decision = risk_enforcer.evaluate(
                                        portfolio_equity=portfolio_equity_now,
                                        current_risk_dollars=running_risk_dollars,
                                        proposed_risk_dollars=proposed_risk,
                                    )
                                    if not risk_decision.admitted:
                                        intent_dollars = existing_dollars * (1 if intent_dollars > 0 else -1)
                                        proposed_dollars = existing_dollars
                                    else:
                                        running_risk_dollars += proposed_risk

                        # Update running totals for symbols processed so far.
                        if proposed_dollars > existing_dollars:
                            approved_additional = proposed_dollars - existing_dollars
                            running_deployed_total += approved_additional
                            sec = sectors[s]
                            running_deployed_per_sector[sec] = (
                                running_deployed_per_sector.get(sec, 0.0) + approved_additional
                            )

                    target_qty = int(intent_dollars / close_px) if (intent_dollars and close_px > 0) else 0
                    delta = target_qty - positions[s].qty
                    if abs(delta) > 1e-9:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        pending_signal[s] = Order(
                            symbol=s, side=side, qty=abs(delta), order_type=OrderType.MARKET,
                            timestamp=next_ts,
                        )

                    # Stop-order scheduling (preserved from Task 26).
                    if s in ts_states and ts_states[s].phase is not TSPhase.DISARMED and target_qty != 0:
                        sgn = 1 if target_qty > 0 else -1
                        stop_px = ts_states[s].stop_price(sign=sgn, bar_idx=i + 1)
                        if stop_px is not None:
                            stop_side = OrderSide.SELL if sgn > 0 else OrderSide.BUY
                            pending_stop[s] = Order(
                                symbol=s, side=stop_side, qty=abs(target_qty),
                                order_type=OrderType.STOP, stop_price=stop_px, timestamp=next_ts,
                            )
                    elif s in ts_states and target_qty == 0:
                        pending_stop[s] = None

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
