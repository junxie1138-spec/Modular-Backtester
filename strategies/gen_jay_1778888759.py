from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 20  # lookback for range-peak and drawdown
    hold_bars: int = 18     # bars to hold after entry (~3-4 weeks)


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778888759"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling(range_window) needs range_window bars; shift(1) needs 1 more
        return params.range_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        range_ratio = (data["high"] - data["low"]) / data["close"]

        # Rolling maximum of range: the recent high-tide mark for volatility
        range_peak = range_ratio.rolling(
            params.range_window, min_periods=params.range_window
        ).max()

        # Range decay: fraction of recent peak; near 0 = maximum tidal ebb
        # Fill NaN during warmup with 1.0 so no spurious ebb signals fire early
        ind["range_decay"] = (range_ratio / range_peak).fillna(1.0)

        # Price drawdown from rolling high (negative = below high)
        rolling_high = data["close"].rolling(
            params.range_window, min_periods=params.range_window
        ).max()
        ind["drawdown"] = ((data["close"] - rolling_high) / rolling_high).fillna(0.0)

        # First day of range re-expansion: tide just began turning back in
        ind["range_expanding"] = (
            (range_ratio > range_ratio.shift(1)).fillna(False).astype(int)
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        in_drawdown = indicators["drawdown"] < -0.02
        range_at_ebb = indicators["range_decay"] < 0.45
        tide_turning = indicators["range_expanding"] == 1

        entry_arr = (in_drawdown & range_at_ebb & tide_turning).astype(int).values

        # Fixed-bar exit: path-dependent, requires a loop
        signal_arr = np.zeros(len(df), dtype=int)
        in_position = False
        entry_bar = -1

        for i in range(len(signal_arr)):
            if not in_position:
                if entry_arr[i] == 1:
                    in_position = True
                    entry_bar = i
                    signal_arr[i] = 1
            else:
                bars_held = i - entry_bar
                if bars_held >= params.hold_bars:
                    # Exit fires; do not re-enter on the same bar
                    in_position = False
                    entry_bar = -1
                    # signal_arr[i] stays 0
                else:
                    signal_arr[i] = 1

        df["signal"] = signal_arr

        # MANDATORY one-bar shift: decision on bar N's close fills on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
