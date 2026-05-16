from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSnapParams:
    compression_window: int = 20
    long_vol_window: int = 60
    compression_ratio: float = 0.70
    snap_zscore: float = 1.5
    min_gap_pct: float = 0.002
    profit_target_pct: float = 0.04
    time_stop_bars: int = 10


class GeneratedStrategy(BaseStrategy["GapSnapParams"]):
    strategy_id = "gen_jay_1778913874"

    @classmethod
    def params_type(cls):
        return GapSnapParams

    @staticmethod
    def warmup_bars(params: GapSnapParams) -> int:
        return params.long_vol_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapSnapParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prior_close = data["close"].shift(1)
        gap = (data["open"] - prior_close) / prior_close.replace(0, np.nan)
        ind["gap"] = gap

        gap_vol_short = gap.rolling(params.compression_window).std()
        gap_vol_long = gap.rolling(params.long_vol_window).std()
        gap_mean_short = gap.rolling(params.compression_window).mean()

        ind["gap_vol_short"] = gap_vol_short
        ind["gap_vol_long"] = gap_vol_long
        ind["is_compressed"] = (
            gap_vol_short < params.compression_ratio * gap_vol_long
        ).astype(int)
        ind["gap_zscore"] = (gap - gap_mean_short) / gap_vol_short.replace(0, np.nan)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSnapParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        gap = indicators["gap"]
        is_compressed = indicators["is_compressed"]
        gap_zscore = indicators["gap_zscore"]

        # Bar N-1: gap-up snaps above the compressed gap distribution
        arm = (
            (gap > params.min_gap_pct)
            & (gap_zscore > params.snap_zscore)
            & (is_compressed == 1)
        )

        # Bar N: prior bar armed AND today closes bullish (two-bar confirmation)
        arm_prev = arm.shift(1).fillna(False)
        confirm = arm_prev & (data["close"] > data["open"])

        raw = confirm.values.astype(bool)
        close = data["close"].values
        n = len(df)

        signal_arr = np.zeros(n, dtype=int)
        entry_bar = -1
        entry_close = 0.0

        for i in range(n):
            in_pos = entry_bar >= 0
            if in_pos:
                bars_held = i - entry_bar
                ret = (close[i] - entry_close) / entry_close
                if ret >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    signal_arr[i] = 0
                    entry_bar = -1
                    entry_close = 0.0
                else:
                    signal_arr[i] = 1
            elif raw[i]:
                signal_arr[i] = 1
                entry_bar = i
                entry_close = close[i]

        df["signal"] = signal_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
