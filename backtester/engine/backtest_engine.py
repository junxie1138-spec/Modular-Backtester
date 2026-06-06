from __future__ import annotations

from dataclasses import asdict, is_dataclass
import pandas as pd

from backtester.analytics.metrics import compute_summary_metrics
from backtester.core.types import BacktestResult, StrategyContext
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator


class BacktestEngine:
    def __init__(self, broker: Broker, portfolio: PortfolioSimulator):
        self.broker = broker
        self.portfolio = portfolio

    def run(self, strategy, data: pd.DataFrame, params, symbol: str, timeframe: str) -> BacktestResult:
        strategy.validate(data, params)

        ctx = StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=strategy.warmup_bars(params),
            metadata={"params": asdict(params) if is_dataclass(params) else {}},
        )

        indicators = strategy.indicators(data, params)
        signal_frame = strategy.generate_signals(data, indicators, ctx, params)

        trades, positions, equity_curve = self.portfolio.simulate(
            data=data,
            signal_frame=signal_frame,
            broker=self.broker,
        )

        summary = compute_summary_metrics(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
            timeframe=timeframe,
        )
        summary["params"] = ctx.metadata["params"]
        summary["symbol"] = symbol
        summary["timeframe"] = timeframe

        return BacktestResult(
            summary=summary,
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
        )
