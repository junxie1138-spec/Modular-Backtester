from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_window: int = 252
    streak_pctl: float = 0.90
    min_streak: int = 2
    vol_window: int = 20
    vol_target: float = 0.15
    size_cap: float = 1.5
    size_floor: float = 0.25


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_a1_1778914273"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(params.streak_window + params.vol_window + 5)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"]
        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        # A bar is an 'up rung' when its close clears the prior bar's full range,
        # a 'down rung' when its close drops below the prior bar's full range.
        rung_up = (close > prev_high).fillna(False).astype(int)
        rung_down = (close < prev_low).fillna(False).astype(int)

        # Consecutive-streak counts via group-reset cumsum (NaN-safe, vectorised).
        up_streak = rung_up.groupby((rung_up == 0).cumsum()).cumsum()
        down_streak = rung_down.groupby((rung_down == 0).cumsum()).cumsum()
        up_streak = up_streak.astype(float)
        down_streak = down_streak.astype(float)

        # Percentile threshold instead of a fixed streak level: each side's
        # streak must exceed its own rolling percentile to count as a breakout.
        w = int(params.streak_window)
        up_thr = up_streak.rolling(w, min_periods=w).quantile(params.streak_pctl)
        dn_thr = down_streak.rolling(w, min_periods=w).quantile(params.streak_pctl)

        # Inverse-volatility sizing toward an annualised vol target.
        ret = close.pct_change()
        rv = ret.rolling(params.vol_window, min_periods=params.vol_window).std()
        rv = rv * np.sqrt(252.0)
        size_raw = params.vol_target / rv
        size_raw = size_raw.replace([np.inf, -np.inf], np.nan)
        size_raw = size_raw.clip(lower=params.size_floor, upper=params.size_cap)
        size_raw = size_raw.fillna(1.0)

        ind = pd.DataFrame(index=data.index)
        ind["up_streak"] = up_streak
        ind["down_streak"] = down_streak
        ind["up_thr"] = up_thr
        ind["dn_thr"] = dn_thr
        ind["size_raw"] = size_raw
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        n = len(data)
        up_s = indicators["up_streak"].to_numpy(dtype=float)
        dn_s = indicators["down_streak"].to_numpy(dtype=float)
        up_t = indicators["up_thr"].to_numpy(dtype=float)
        dn_t = indicators["dn_thr"].to_numpy(dtype=float)

        ms = float(params.min_streak)
        sig = np.zeros(n, dtype=int)
        pos = 0

        for i in range(n):
            ut = up_t[i]
            dt = dn_t[i]
            long_thr = ms if np.isnan(ut) else max(ms, ut)
            short_thr = ms if np.isnan(dt) else max(ms, dt)

            long_entry = (up_s[i] > 0.0) and (up_s[i] >= long_thr)
            short_entry = (dn_s[i] > 0.0) and (dn_s[i] >= short_thr)

            if pos == 0:
                if long_entry:
                    pos = 1
                elif short_entry:
                    pos = -1
            elif pos == 1:
                # Signal-reversal exit: leave the long only when the short
                # entry condition flips on - then reverse straight into it.
                if short_entry:
                    pos = -1
            else:  # pos == -1
                if long_entry:
                    pos = 1

            sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        # Mandatory one-bar shift: decide on bar N's close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size_raw"].shift(1).fillna(1.0)
        size = size.clip(lower=params.size_floor)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
