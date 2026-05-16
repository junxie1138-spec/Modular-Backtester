from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compression_window: int = 20
    compression_threshold: float = 0.75
    min_streak: int = 5
    breakout_window: int = 20
    atr_window: int = 14
    trail_k: float = 2.0
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778903887"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return (
            max(params.compression_window, params.breakout_window + 1)
            + params.min_streak
            + params.atr_window * 2
            + 2
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.ewm(span=params.atr_window, adjust=False).mean()

        bar_range = high - low
        med_range = bar_range.rolling(
            params.compression_window, min_periods=params.compression_window
        ).median()

        # Elastic compression flag: range < threshold * rolling-median range
        compressed = (bar_range < params.compression_threshold * med_range).astype(float)
        compressed = compressed.where(med_range.notna(), np.nan)

        # Vectorised consecutive compression streak via cumsum-group trick
        is_not_compressed = compressed != 1.0  # True for 0 and NaN
        group = is_not_compressed.cumsum()
        streak = compressed.fillna(0.0).groupby(group).cumsum()
        streak = streak.where(~is_not_compressed, 0.0)
        streak = streak.where(compressed.notna(), np.nan)
        ind["compression_streak"] = streak

        # Plastic deformation gate: close > rolling max of *prior* closes (no lookahead)
        prior_max = close.shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        ind["plastic_breakout"] = (
            (close > prior_max).astype(float).where(prior_max.notna(), np.nan)
        )

        ind["close"] = close
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = indicators["close"].to_numpy(dtype=float)
        atr_arr = indicators["atr"].to_numpy(dtype=float)
        streak_arr = indicators["compression_streak"].to_numpy(dtype=float)
        breakout_arr = indicators["plastic_breakout"].to_numpy(dtype=float)

        n = len(close_arr)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size)

        in_position = False
        hwm = np.nan

        for i in range(n):
            c = close_arr[i]
            a = atr_arr[i]
            streak = streak_arr[i]
            breakout = breakout_arr[i]

            if np.isnan(c) or np.isnan(a) or np.isnan(streak) or np.isnan(breakout):
                signal[i] = 0
                continue

            if in_position:
                if c > hwm:
                    hwm = c
                # Trailing stop: only ratchets up; exit when close falls k*ATR below HWM
                if c <= hwm - params.trail_k * a:
                    signal[i] = 0
                    in_position = False
                    hwm = np.nan
                else:
                    signal[i] = 1
            else:
                # AND gate: both primitives must agree
                # Primitive 1 (elastic coil): >= min_streak consecutive compressed bars
                # Primitive 2 (plastic breakout): close above rolling prior-max
                if streak >= params.min_streak and breakout == 1.0:
                    signal[i] = 1
                    in_position = True
                    hwm = c
                else:
                    signal[i] = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        # Shift: decide on bar N close, execute on bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
