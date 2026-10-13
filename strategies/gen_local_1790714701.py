from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    percentile_window: int = 126
    area_percentile: float = 0.8
    path_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714701"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return params.percentile_window + params.path_bars

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        x = np.log(close / close.shift(1))
        span = (data["high"] - data["low"]).replace(0, np.nan)
        y = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / span
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Signed second-order path area across the latest four daily increments.
        xs = [x.shift(i) for i in (3, 2, 1, 0)]
        ys = [y.shift(i) for i in (3, 2, 1, 0)]
        area = pd.Series(0.0, index=data.index)
        for i in range(4):
            for j in range(i + 1, 4):
                area = area + xs[i] * ys[j] - ys[i] * xs[j]
        area = area.replace([np.inf, -np.inf], np.nan)
        threshold = area.abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.area_percentile)
        return pd.DataFrame({"area": area, "threshold": threshold}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        area = indicators["area"]
        threshold = indicators["threshold"]
        long_entry = (area > 0) & (area.abs() >= threshold) & threshold.notna()
        short_entry = (area < 0) & (area.abs() >= threshold) & threshold.notna()

        # Hold each direction until the opposite percentile setup appears.
        position = np.zeros(len(data), dtype=int)
        long_values = long_entry.fillna(False).to_numpy()
        short_values = short_entry.fillna(False).to_numpy()
        state = 0
        for i in range(len(data)):
            if state == 1 and short_values[i]:
                state = 0
            elif state == -1 and long_values[i]:
                state = 0
            elif state == 0:
                if long_values[i]:
                    state = 1
                elif short_values[i]:
                    state = -1
            position[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(position, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
