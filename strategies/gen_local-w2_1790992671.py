from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    ma_window: int = 4
    compression_ratio: float = 0.8


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = "gen_local-w2_1790992671"

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return int(max(21, params.ma_window))

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        normalized_range = (data["high"] - data["low"]) / close.where(close > 0)
        recent_range = normalized_range.rolling(3, min_periods=3).mean()
        reference_range = normalized_range.rolling(20, min_periods=20).mean().shift(1)
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        deviation = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        gap_return = (data["open"] - previous_close) / previous_close.where(previous_close > 0)
        gap_pressure = gap_return.rolling(3, min_periods=3).sum()
        result = pd.DataFrame(index=data.index)
        result["normalized_range"] = normalized_range
        result["recent_range"] = recent_range
        result["reference_range"] = reference_range
        result["compression"] = recent_range / reference_range.where(reference_range > 0)
        result["mean"] = mean
        result["deviation"] = deviation
        result["zscore"] = (close - mean) / deviation.where(deviation > 0)
        result["gap_return"] = gap_return
        result["gap_pressure"] = gap_pressure
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCompressionParams) -> SignalFrame:
        valid = indicators[["compression", "zscore", "gap_pressure"]].notna().all(axis=1)
        compressed = valid & (indicators["compression"] <= params.compression_ratio)
        long_entry = compressed & (indicators["zscore"] >= 0.75) & (indicators["gap_pressure"] > 0)
        short_entry = compressed & (indicators["zscore"] <= -0.75) & (indicators["gap_pressure"] < 0)
        entry_direction = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry_direction):
            direction = int(direction)
            if position == 0:
                position = direction
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
