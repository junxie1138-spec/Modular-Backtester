from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAlignmentParams:
    close_streak_min: int = 3
    high_streak_min: int = 2
    atr_window: int = 14
    spike_atr_mult: float = 2.0
    refractory_bars: int = 5
    trail_atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[StreakAlignmentParams]):
    strategy_id = "gen_a1_1779652797"

    @classmethod
    def params_type(cls):
        return StreakAlignmentParams

    def warmup_bars(self, params: StreakAlignmentParams) -> int:
        return int(
            params.atr_window
            + max(params.close_streak_min, params.high_streak_min)
            + params.refractory_bars
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: StreakAlignmentParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        up = (close.diff() > 0).astype(int)
        not_up = (up == 0).astype(int)
        group_up = not_up.cumsum()
        close_streak = up.groupby(group_up).cumsum()

        hh = (high.diff() > 0).astype(int)
        not_hh = (hh == 0).astype(int)
        group_hh = not_hh.cumsum()
        high_streak = hh.groupby(group_hh).cumsum()

        bar_change = close.diff().abs()
        spike_threshold = params.spike_atr_mult * atr
        is_spike = (bar_change > spike_threshold).fillna(False).astype(int)

        refractory = (
            is_spike.rolling(params.refractory_bars, min_periods=1).max().fillna(0).astype(int)
        )

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["close_streak"] = close_streak.astype(float)
        out["high_streak"] = high_streak.astype(float)
        out["is_spike"] = is_spike
        out["refractory"] = refractory
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakAlignmentParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close_streak = indicators["close_streak"].to_numpy(dtype=float)
        high_streak = indicators["high_streak"].to_numpy(dtype=float)
        refractory = indicators["refractory"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=int)
        in_pos = False
        hwm = np.nan

        cs_min = float(params.close_streak_min)
        hs_min = float(params.high_streak_min)
        trail_mult = float(params.trail_atr_mult)

        for i in range(n):
            a = atr[i]
            if np.isnan(a) or np.isnan(close[i]):
                if in_pos:
                    raw[i] = 1
                else:
                    raw[i] = 0
                continue

            if not in_pos:
                cond_streaks = (close_streak[i] >= cs_min) and (high_streak[i] >= hs_min)
                cond_no_refractory = refractory[i] < 0.5
                if cond_streaks and cond_no_refractory:
                    in_pos = True
                    hwm = close[i]
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                if close[i] > hwm:
                    hwm = close[i]
                stop_level = hwm - trail_mult * a
                if close[i] < stop_level:
                    in_pos = False
                    hwm = np.nan
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
