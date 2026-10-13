from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    area_threshold: float = 0.025
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791301571"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        gap = np.log(data["open"] / data["close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
        day = np.log(data["close"] / data["open"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        x, y = gap.cumsum(), day.cumsum()
        cross = (x - gap) * day - (y - day) * gap
        x0, y0 = x.shift(5).fillna(0), y.shift(5).fillna(0)
        area = 0.5 * (cross.rolling(5).sum() - x0 * (y - y0) + y0 * (x - x0))
        scale = (gap.abs() + day.abs()).rolling(5).sum().pow(2)
        out = pd.DataFrame(index=data.index)
        out["area"] = area.div(scale.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        area = indicators["area"].to_numpy()
        weekday = data.index.dayofweek.to_numpy()
        direction = 0
        entered = -1
        for i in range(len(data)):
            if direction:
                if i - entered >= params.hold_bars:
                    direction = 0
            elif i >= self.warmup_bars(params) and weekday[i] == 4 and abs(area[i]) >= params.area_threshold:
                direction = -1 if area[i] > 0 else 1
                entered = i
            raw[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
