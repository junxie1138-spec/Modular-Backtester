from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    slope_span: int = 4
    vol_window: int = 20
    history: int = 63


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791224551"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return params.history + params.vol_window + params.slope_span + 3

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        log_close = np.log(data["close"].where(data["close"] > 0))
        volatility = log_close.diff().rolling(params.vol_window).std().replace(0, np.nan)
        slope = (log_close - log_close.shift(params.slope_span)) / params.slope_span
        innovation = slope.diff() / volatility
        shape = pd.DataFrame({
            "now": innovation,
            "prior": innovation.shift(1),
            "older": innovation.shift(2),
            "strain": slope / volatility,
        }, index=data.index)

        distances = []
        outcomes = []
        for lag in range(3, params.history + 1):
            difference = shape - shape.shift(lag)
            distance = difference[["now", "prior", "older"]].pow(2).sum(axis=1, min_count=3)
            distance += 0.25 * difference["strain"].pow(2)
            outcome = np.sign(np.log(data["close"].shift(lag - 2) / data["close"].shift(lag)))
            distances.append(distance.to_numpy())
            outcomes.append(outcome.to_numpy())

        distance_array = np.column_stack(distances)
        outcome_array = np.column_stack(outcomes)
        distance_array = np.where(
            np.isfinite(distance_array) & np.isfinite(outcome_array), distance_array, np.inf
        )
        nearest = np.argpartition(distance_array, 4, axis=1)[:, :5]
        valid = np.isfinite(np.take_along_axis(distance_array, nearest, axis=1)).all(axis=1)
        vote = np.take_along_axis(outcome_array, nearest, axis=1).mean(axis=1)
        vote = np.where(valid, vote, 0.0)
        vote[:self.warmup_bars(params)] = 0.0
        return pd.DataFrame({"vote": vote}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        votes = indicators["vote"].fillna(0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, vote in enumerate(votes):
            direction = int(np.sign(vote))
            if position == 0:
                position = direction
            elif direction == -position:
                position = 0
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(0.2 + 0.8 * np.abs(votes), index=data.index).shift(1).fillna(0.2)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
