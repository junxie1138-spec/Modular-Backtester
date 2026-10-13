from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShiftParams:
    rank_window: int = 40
    tail: float = 0.85
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeShiftParams]):
    strategy_id = "gen_local_1791225029"

    @classmethod
    def params_type(cls) -> type[RangeShiftParams]:
        return RangeShiftParams

    @classmethod
    def warmup_bars(cls, params: RangeShiftParams) -> int:
        return max(200, params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: RangeShiftParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        previous_midpoint = (data["high"].shift(1) + data["low"].shift(1)) / 2
        midpoint = (data["high"] + data["low"]) / 2
        combined_range = (data["high"] - data["low"]) + (data["high"].shift(1) - data["low"].shift(1))
        shift = (midpoint - previous_midpoint) / combined_range.replace(0, np.nan)
        out["range_shift"] = shift
        out["upper"] = shift.rolling(params.rank_window, min_periods=params.rank_window).quantile(params.tail)
        out["lower"] = shift.rolling(params.rank_window, min_periods=params.rank_window).quantile(1 - params.tail)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeShiftParams,
    ) -> SignalFrame:
        ready = np.arange(len(data)) >= self.warmup_bars(params)
        long_entry = (ready & (data["close"] > indicators["ma200"]) &
                      (data["close"] > data["open"]) &
                      (indicators["range_shift"] > indicators["upper"])).to_numpy()
        short_entry = (ready & (data["close"] < indicators["ma200"]) &
                       (data["close"] < data["open"]) &
                       (indicators["range_shift"] < indicators["lower"])).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        for i in range(len(data)):
            if position:
                age += 1
                if age >= params.hold_bars:
                    position = 0
            elif long_entry[i]:
                position = 1
                age = 0
            elif short_entry[i]:
                position = -1
                age = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
