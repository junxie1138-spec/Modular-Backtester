from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_short: int = 5
    atr_long: int = 30
    compress_thresh: float = 0.80
    close_pos_window: int = 10
    close_pos_thresh: float = 0.55
    trail_k: float = 2.0
    atr_trail: int = 14


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778892676"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.atr_long, params.close_pos_window, params.atr_trail) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_short = tr.rolling(params.atr_short).mean()
        atr_long = tr.rolling(params.atr_long).mean()
        ind["atr_ratio"] = atr_short / (atr_long + 1e-8)
        ind["atr_trail"] = tr.rolling(params.atr_trail).mean()

        hl_range = high - low
        close_pos = (close - low) / (hl_range + 1e-8)
        close_pos = close_pos.clip(0.0, 1.0)
        ind["close_pos_mean"] = close_pos.rolling(params.close_pos_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        atr_ratio = indicators["atr_ratio"]
        atr_trail = indicators["atr_trail"]
        close_pos_mean = indicators["close_pos_mean"]

        n = len(data)
        signal = np.zeros(n, dtype=int)

        prev_signal = 0
        high_water = np.nan

        for i in range(n):
            c = float(close.iloc[i])
            ar = float(atr_ratio.iloc[i])
            at = float(atr_trail.iloc[i])
            cpm = float(close_pos_mean.iloc[i])

            if np.isnan(ar) or np.isnan(at) or np.isnan(cpm):
                signal[i] = 0
                prev_signal = 0
                high_water = np.nan
                continue

            in_position = prev_signal == 1

            if in_position:
                if np.isnan(high_water) or c > high_water:
                    high_water = c
                if c < high_water - params.trail_k * at:
                    signal[i] = 0
                    high_water = np.nan
                else:
                    signal[i] = 1
            else:
                compressed = ar < params.compress_thresh
                bullish_close = cpm > params.close_pos_thresh

                if compressed and bullish_close:
                    signal[i] = 1
                    high_water = c
                else:
                    signal[i] = 0

            prev_signal = signal[i]

        df = pd.DataFrame(
            {"signal": signal, "size": np.ones(n, dtype=float)},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
