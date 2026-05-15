from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    gap_min_ratio: float = 0.10
    gap_cap_ratio: float = 0.80
    range_window: int = 5
    compression_short: int = 5
    compression_long: int = 20
    compression_thresh: float = 1.05
    ma_window: int = 200
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_a1_1778886084"

    @classmethod
    def params_type(cls):
        return GapCapacityParams

    @staticmethod
    def warmup_bars(params: GapCapacityParams) -> int:
        longest = max(
            int(params.ma_window),
            int(params.compression_long),
            int(params.range_window),
        )
        return longest + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        rng = (data["high"] - data["low"]).astype(float)

        # Capacity buffer: smoothed prior high-low range up to bar t-1.
        typical_range = rng.rolling(int(params.range_window)).mean().shift(1)
        typical_range = typical_range.replace(0.0, np.nan)

        gap = data["open"].astype(float) - data["close"].shift(1).astype(float)
        gap_ratio = gap / typical_range

        short_range = rng.rolling(int(params.compression_short)).mean()
        long_range = rng.rolling(int(params.compression_long)).mean()
        long_range = long_range.replace(0.0, np.nan)
        compression = short_range / long_range

        ma = data["close"].rolling(int(params.ma_window)).mean()

        out["gap_ratio"] = gap_ratio
        out["compression"] = compression
        out["ma"] = ma

        within_capacity = (gap_ratio >= float(params.gap_min_ratio)) & (
            gap_ratio <= float(params.gap_cap_ratio)
        )
        compressed = compression < float(params.compression_thresh)
        regime = data["close"] > ma

        entry = within_capacity & compressed & regime
        out["entry"] = entry.fillna(False).astype(bool)

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCapacityParams,
    ) -> SignalFrame:
        n = len(data)
        entry = indicators["entry"].to_numpy(dtype=bool)
        hold_bars = max(1, int(params.hold_bars))

        target = np.zeros(n, dtype=np.int64)
        position = 0
        bars_held = 0

        for i in range(n):
            if position == 1:
                target[i] = 1
                bars_held += 1
                if bars_held >= hold_bars:
                    position = 0
                    bars_held = 0
            elif entry[i]:
                position = 1
                bars_held = 1
                target[i] = 1
                if bars_held >= hold_bars:
                    position = 0
                    bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
