from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HigherLowStreakParams:
    streak_min: int = 3
    range_window: int = 20
    range_upper_pct: float = 0.70
    spike_window: int = 10
    spike_z_thresh: float = 2.0
    spike_lookback: int = 50
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[HigherLowStreakParams]):
    strategy_id = "gen_a1_1779883960"

    @classmethod
    def params_type(cls):
        return HigherLowStreakParams

    @classmethod
    def warmup_bars(cls, params: HigherLowStreakParams) -> int:
        return int(max(params.range_window, params.spike_lookback) + params.spike_window + 5)

    def indicators(self, data: pd.DataFrame, params: HigherLowStreakParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Primitive 1: consecutive higher-low streak count (structural uptrend geometry)
        hl_up = (low > low.shift(1)).astype(int)
        reset_grp = (hl_up == 0).cumsum()
        streak = hl_up.groupby(reset_grp).cumsum()

        # Primitive 2: positional strength - where close sits in rolling N-day range
        roll_low = low.rolling(params.range_window, min_periods=params.range_window).min()
        roll_high = high.rolling(params.range_window, min_periods=params.range_window).max()
        rng = (roll_high - roll_low).replace(0, np.nan)
        range_pos = (close - roll_low) / rng

        # Refractory gate: true-range z-score spike recency
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr_mean = tr.rolling(params.spike_lookback, min_periods=params.spike_lookback).mean()
        tr_std = tr.rolling(params.spike_lookback, min_periods=params.spike_lookback).std()
        tr_z = (tr - tr_mean) / tr_std.replace(0, np.nan)
        spiked_bar = (tr_z >= params.spike_z_thresh).fillna(False).astype(int)
        recent_spike = spiked_bar.rolling(params.spike_window, min_periods=1).max().fillna(0)

        out = pd.DataFrame(index=data.index)
        out["streak"] = streak.astype(float)
        out["range_pos"] = range_pos
        out["recent_spike"] = recent_spike.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HigherLowStreakParams,
    ) -> SignalFrame:
        streak = indicators["streak"]
        range_pos = indicators["range_pos"]
        recent_spike = indicators["recent_spike"]

        primitive_a = streak.fillna(0) >= float(params.streak_min)
        primitive_b = range_pos.fillna(-1.0) >= float(params.range_upper_pct)
        no_spike = recent_spike.fillna(0) <= 0

        entry_mask = (primitive_a & primitive_b & no_spike).to_numpy()

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        hold = max(1, int(params.hold_bars))

        i = 0
        while i < n:
            if entry_mask[i]:
                end = min(n, i + hold)
                for j in range(i, end):
                    raw_signal[j] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
