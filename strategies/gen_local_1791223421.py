from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueRecoveryParams:
    drawdown_window: int = 21
    capacity_threshold: float = 0.65


class GeneratedStrategy(BaseStrategy[QueueRecoveryParams]):
    strategy_id = "gen_local_1791223421"

    @classmethod
    def params_type(cls) -> type[QueueRecoveryParams]:
        return QueueRecoveryParams

    @classmethod
    def warmup_bars(cls, params: QueueRecoveryParams) -> int:
        return params.drawdown_window + 1

    def indicators(
        self, data: pd.DataFrame, params: QueueRecoveryParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change()
        prior_peak = close.shift(1).rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max()
        queued_drawdown = (1.0 - close / prior_peak).clip(lower=0.0)
        noise_capacity = returns.rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).std() * np.sqrt(3.0)
        capacity = pd.concat([queued_drawdown, noise_capacity], axis=1).max(axis=1)
        recovery_throughput = close.pct_change(3)
        utilization = recovery_throughput.div(capacity.replace(0.0, np.nan))

        return pd.DataFrame(
            {
                "queued_drawdown": queued_drawdown,
                "recovery_throughput": recovery_throughput,
                "utilization": utilization.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QueueRecoveryParams,
    ) -> SignalFrame:
        utilization = indicators["utilization"]
        long_entry = utilization >= params.capacity_threshold
        short_entry = utilization <= -params.capacity_threshold

        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 0:
                if long_entry.iat[i]:
                    position = 1
                elif short_entry.iat[i]:
                    position = -1
            elif position == 1 and short_entry.iat[i]:
                position = 0
            elif position == -1 and long_entry.iat[i]:
                position = 0
            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(positions, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
