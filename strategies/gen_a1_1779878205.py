from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStreakParams:
    min_streak: int = 3
    max_streak: int = 8
    snr_window: int = 20
    snr_threshold: float = 0.5
    breakeven_pct: float = 0.03
    atr_window: int = 14
    atr_mult: float = 2.5
    max_hold_bars: int = 18
    size_base: float = 0.4
    size_per_streak: float = 0.15
    size_cap: float = 1.0
    trend_ma: int = 200


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779878205"

    @classmethod
    def params_type(cls):
        return GapStreakParams

    def warmup_bars(self, params):
        return int(max(params.trend_ma, params.snr_window, params.atr_window)) + 5

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        prev_close = close.shift(1)

        gap = ((data["open"] - prev_close) / prev_close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["gap"] = gap

        up_gap = (gap > 0).astype(int)
        grp = (up_gap != up_gap.shift(1)).cumsum()
        streak_raw = up_gap.groupby(grp).cumsum()
        out["up_streak"] = (streak_raw * up_gap).astype(int)

        gap_mean = gap.rolling(params.snr_window, min_periods=params.snr_window).mean()
        gap_std = gap.rolling(params.snr_window, min_periods=params.snr_window).std()
        snr = gap_mean / gap_std.replace(0, np.nan)
        out["snr"] = snr.replace([np.inf, -np.inf], np.nan)

        high = data["high"]
        low = data["low"]
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        out["trend_ma"] = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()

        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        streak = indicators["up_streak"].to_numpy(dtype=float)
        snr = indicators["snr"].to_numpy(dtype=float)
        trend_ma = indicators["trend_ma"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.zeros(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_hit = False
        bars_held = 0
        entry_size = 0.0

        for i in range(n):
            atr_i = atr[i]
            snr_i = snr[i]
            ma_i = trend_ma[i]
            streak_i = streak[i]
            close_i = close[i]

            if not in_pos:
                if (
                    not np.isnan(snr_i)
                    and not np.isnan(ma_i)
                    and not np.isnan(atr_i)
                    and atr_i > 0.0
                    and streak_i >= params.min_streak
                    and snr_i >= params.snr_threshold
                    and close_i > ma_i
                ):
                    streak_capped = min(streak_i, float(params.max_streak))
                    if params.snr_threshold > 0.0:
                        snr_scale = snr_i / params.snr_threshold
                    else:
                        snr_scale = 1.0
                    if snr_scale < 1.0:
                        snr_scale = 1.0
                    if snr_scale > 2.0:
                        snr_scale = 2.0
                    raw_size = (
                        params.size_base
                        + params.size_per_streak * (streak_capped - params.min_streak)
                    ) * snr_scale
                    if raw_size > params.size_cap:
                        raw_size = params.size_cap
                    if raw_size < 0.05:
                        raw_size = 0.05

                    signal[i] = 1
                    size[i] = raw_size
                    in_pos = True
                    entry_price = close_i
                    stop = close_i - params.atr_mult * atr_i
                    breakeven_hit = False
                    bars_held = 0
                    entry_size = raw_size
            else:
                bars_held += 1
                if not breakeven_hit:
                    if close_i >= entry_price * (1.0 + params.breakeven_pct):
                        breakeven_hit = True
                        if entry_price > stop:
                            stop = entry_price
                else:
                    if not np.isnan(atr_i):
                        new_stop = close_i - params.atr_mult * atr_i
                        if new_stop > stop:
                            stop = new_stop

                exit_now = False
                if close_i <= stop:
                    exit_now = True
                if bars_held >= params.max_hold_bars:
                    exit_now = True

                if exit_now:
                    signal[i] = 0
                    size[i] = 0.0
                    in_pos = False
                    entry_price = 0.0
                    stop = 0.0
                    breakeven_hit = False
                    bars_held = 0
                    entry_size = 0.0
                else:
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.0)
        df.loc[df["signal"] == 0, "size"] = 1.0
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
