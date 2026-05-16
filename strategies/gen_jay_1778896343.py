from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeFractalParams:
    short_window: int = 5
    medium_window: int = 21
    rank_window: int = 63
    arm_threshold: float = 0.30
    fire_threshold: float = 0.75
    hold_bars: int = 15
    close_pos_min: float = 0.50
    min_size: float = 0.30
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[RangeFractalParams]):
    strategy_id = "gen_jay_1778896343"

    @classmethod
    def params_type(cls) -> type[RangeFractalParams]:
        return RangeFractalParams

    @staticmethod
    def warmup_bars(params: RangeFractalParams) -> int:
        return params.rank_window + params.medium_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeFractalParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        short_high = data["high"].rolling(params.short_window).max()
        short_low = data["low"].rolling(params.short_window).min()
        short_range = short_high - short_low

        medium_high = data["high"].rolling(params.medium_window).max()
        medium_low = data["low"].rolling(params.medium_window).min()
        medium_range = medium_high - medium_low

        safe_medium = medium_range.where(medium_range > 0, np.nan)
        ind["fractal_ratio"] = short_range / safe_medium

        ind["ratio_rank"] = ind["fractal_ratio"].rolling(params.rank_window).rank(pct=True)

        bar_range = (data["high"] - data["low"]).where(
            data["high"] > data["low"], np.nan
        )
        ind["close_pos"] = (data["close"] - data["low"]) / bar_range

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeFractalParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=np.int64)
        sizes = np.full(n, 0.5)

        ratio_rank = indicators["ratio_rank"].values
        close_pos = indicators["close_pos"].values

        armed = False
        bars_remaining = 0
        entry_size = 0.5

        for i in range(n):
            rr = ratio_rank[i]

            if bars_remaining > 0:
                signals[i] = 1
                sizes[i] = entry_size
                bars_remaining -= 1
            else:
                if not np.isnan(rr) and rr < params.arm_threshold:
                    armed = True

                cp = close_pos[i]
                if (
                    armed
                    and not np.isnan(rr)
                    and rr > params.fire_threshold
                    and not np.isnan(cp)
                    and cp >= params.close_pos_min
                ):
                    signals[i] = 1
                    entry_size = params.min_size + (
                        params.max_size - params.min_size
                    ) * rr
                    sizes[i] = entry_size
                    bars_remaining = params.hold_bars - 1
                    armed = False

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.5)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
