from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback_weeks: int = 8
    streak_threshold: int = 2
    snr_min: float = 0.0003
    size: float = 1.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778910051"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.lookback_weeks * 5 + params.streak_threshold + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        ret = data["close"].pct_change()

        # Per-DOW rolling mean return: seasonal baseline for each weekday
        dow_avg = pd.Series(np.nan, index=data.index)
        for dow in range(5):
            mask = data.index.dayofweek == dow
            dow_ret = ret[mask]
            rolling_avg = dow_ret.rolling(params.lookback_weeks, min_periods=3).mean()
            dow_avg[mask] = rolling_avg.values

        # Excess return vs the DOW-specific seasonal baseline
        excess = ret - dow_avg

        # Direction of excess: +1 beating baseline, -1 missing it, 0 flat or NaN
        excess_dir = np.sign(excess).fillna(0).astype(int)

        # Consecutive streak of same excess direction (vectorised change-point method)
        prev_dir = excess_dir.shift(1).fillna(0).astype(int)
        change = (excess_dir != prev_dir) | (excess_dir == 0) | (prev_dir == 0)
        group_id = change.cumsum()
        cumcnt = excess_dir.groupby(group_id).cumcount() + 1
        cumcnt = cumcnt.where(excess_dir != 0, 0)
        streak = (excess_dir * cumcnt).astype(int)

        ind["streak"] = streak
        ind["dow_avg"] = dow_avg
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        streak = indicators["streak"]
        dow_avg = indicators["dow_avg"]

        # Symmetric entry: long when consecutive DOW-excess beats reach threshold
        # and the DOW baseline itself is bullish; exact mirror for short.
        long_cond = (streak >= params.streak_threshold) & (dow_avg > params.snr_min)
        short_cond = (streak <= -params.streak_threshold) & (dow_avg < -params.snr_min)

        raw_signal = pd.Series(0, index=data.index, dtype=int)
        raw_signal[long_cond] = 1
        raw_signal[short_cond] = -1

        # Signal-reversal exit: hold position until the opposite signal fires
        raw_arr = raw_signal.values
        signal_arr = np.zeros(len(raw_arr), dtype=int)
        pos = 0
        for i in range(len(raw_arr)):
            s = int(raw_arr[i])
            if s == 1:
                pos = 1
            elif s == -1:
                pos = -1
            signal_arr[i] = pos

        df["signal"] = signal_arr
        df["size"] = float(params.size)

        # Mandatory 1-bar shift: decision on bar N, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
