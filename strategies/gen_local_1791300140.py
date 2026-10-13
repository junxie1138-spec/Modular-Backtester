from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_window: int = 9
    degree_threshold: float = 0.15
    phase_days: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791300140"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.graph_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        ma = close.rolling(200, min_periods=200).mean()
        out["ma200"] = ma.fillna(0.0)
        out["ready"] = ma.notna()

        upward = pd.Series(1.0, index=data.index)
        downward = pd.Series(1.0, index=data.index)
        for lag in range(2, params.graph_window + 1):
            prior = close.shift(lag)
            sees_up = prior.notna()
            sees_down = prior.notna()
            for gap in range(1, lag):
                sightline = close + (prior - close) * (gap / lag)
                middle = close.shift(gap)
                sees_up = sees_up & middle.lt(sightline)
                sees_down = sees_down & middle.gt(sightline)
            upward = upward + sees_up.astype(float)
            downward = downward + sees_down.astype(float)

        contrast = (downward - upward) / params.graph_window
        out["graph_score"] = contrast.rolling(3, min_periods=3).mean().fillna(0.0)
        out["month_session"] = pd.Series(1, index=data.index).groupby(data.index.to_period("M")).cumsum()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        seasonal = indicators["month_session"].le(params.phase_days)
        ready = indicators["ready"] & seasonal
        long_entry = ready & data["close"].gt(indicators["ma200"]) & indicators["graph_score"].ge(params.degree_threshold)
        short_entry = ready & data["close"].lt(indicators["ma200"]) & indicators["graph_score"].le(-params.degree_threshold)

        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1 and not long_entry.iat[i]:
                position = 0
            elif position == -1 and not short_entry.iat[i]:
                position = 0
            elif position == 0:
                if long_entry.iat[i]:
                    position = 1
                elif short_entry.iat[i]:
                    position = -1
            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
