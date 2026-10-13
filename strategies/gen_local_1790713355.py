from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    area_window: int = 4
    area_threshold: float = 0.00005
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local_1790713355"

    @classmethod
    def params_type(cls):
        return PathAreaParams

    @classmethod
    def warmup_bars(cls, params: PathAreaParams) -> int:
        return params.area_window + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: PathAreaParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        span = (data["high"] - data["low"]).replace(0, np.nan)
        location = ((data["close"] - data["low"]) / span - 0.5).fillna(0.0)
        area_increment = ret.shift(1) * location - location.shift(1) * ret
        area = area_increment.rolling(params.area_window, min_periods=params.area_window).sum()
        momentum = data["close"].pct_change(3)
        return pd.DataFrame({"area": area, "momentum": momentum}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PathAreaParams,
    ) -> SignalFrame:
        area = indicators["area"].fillna(0.0)
        momentum = indicators["momentum"].fillna(0.0)
        raw = pd.Series(0, index=data.index, dtype=int)
        remaining = 0
        active = 0
        for i in range(len(data)):
            if remaining:
                raw.iloc[i] = active
                remaining -= 1
            else:
                a, m = area.iloc[i], momentum.iloc[i]
                if a > params.area_threshold and m > 0:
                    active, remaining = 1, params.hold_bars - 1
                    raw.iloc[i] = active
                elif a < -params.area_threshold and m < 0:
                    active, remaining = -1, params.hold_bars - 1
                    raw.iloc[i] = active
                else:
                    active = 0
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
