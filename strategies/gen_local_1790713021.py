from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendRangeParams:
    range_lookback: int = 20
    expansion_threshold: float = 1.25
    efficiency_threshold: float = 0.45
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TrendRangeParams]):
    strategy_id = "gen_local_1790713021"

    @classmethod
    def params_type(cls):
        return TrendRangeParams

    @classmethod
    def warmup_bars(cls, params: TrendRangeParams) -> int:
        return max(params.range_lookback + 1, 11)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: TrendRangeParams) -> pd.DataFrame:
        bar_range = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
        range_baseline = bar_range.rolling(params.range_lookback, min_periods=params.range_lookback).mean()
        range_expansion = bar_range / range_baseline.replace(0, np.nan)
        daily_change = data["close"].diff()
        path_length = daily_change.abs().rolling(10, min_periods=10).sum()
        efficiency = daily_change.rolling(10, min_periods=10).sum().abs() / path_length.replace(0, np.nan)
        direction = np.sign(daily_change.rolling(10, min_periods=10).sum())
        return pd.DataFrame(
            {"range_expansion": range_expansion, "efficiency": efficiency, "direction": direction},
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendRangeParams,
    ) -> SignalFrame:
        valid = indicators[["range_expansion", "efficiency", "direction"]].notna().all(axis=1)
        entries = (
            valid
            & (indicators["range_expansion"] >= params.expansion_threshold)
            & (indicators["efficiency"] >= params.efficiency_threshold)
            & indicators["direction"].ne(0)
        )
        intent = np.zeros(len(data), dtype=int)
        active = 0
        remaining = 0
        for i in range(len(data)):
            if remaining > 0:
                intent[i] = active
                remaining -= 1
            elif entries.iloc[i]:
                active = int(indicators["direction"].iloc[i])
                intent[i] = active
                remaining = max(0, params.hold_bars - 1)
            else:
                active = 0
        df = pd.DataFrame({"signal": intent, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
