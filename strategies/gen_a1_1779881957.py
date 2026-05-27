from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAutocorrHysteresisParams:
    autocorr_window: int = 20
    arm_threshold: float = 0.25
    disarm_threshold: float = 0.0
    gap_threshold: float = 0.001
    ma_window: int = 200
    atr_window: int = 14
    atr_mult: float = 3.0
    min_hold_bars: int = 5


class GeneratedStrategy(BaseStrategy[GapAutocorrHysteresisParams]):
    strategy_id = "gen_a1_1779881957"

    @classmethod
    def params_type(cls):
        return GapAutocorrHysteresisParams

    def warmup_bars(self, params):
        return int(max(params.ma_window, params.autocorr_window, params.atr_window)) + 2

    def indicators(self, data, params):
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        gap_ret = (open_ / prev_close) - 1.0
        gap_lag = gap_ret.shift(1)

        w = int(params.autocorr_window)
        mean_x = gap_ret.rolling(w).mean()
        mean_y = gap_lag.rolling(w).mean()
        std_x = gap_ret.rolling(w).std(ddof=0)
        std_y = gap_lag.rolling(w).std(ddof=0)
        mean_xy = (gap_ret * gap_lag).rolling(w).mean()
        cov = mean_xy - (mean_x * mean_y)
        denom = std_x * std_y
        denom = denom.where(denom > 0.0, np.nan)
        gap_autocorr = cov / denom

        ma200 = close.rolling(int(params.ma_window)).mean()

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(int(params.atr_window)).mean()

        ind = pd.DataFrame(
            {
                "gap_ret": gap_ret,
                "gap_autocorr": gap_autocorr,
                "ma200": ma200,
                "atr": atr,
            },
            index=data.index,
        )
        return ind

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        gap_ret = indicators["gap_ret"].to_numpy(dtype=float)
        autocorr = indicators["gap_autocorr"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        armed = False
        in_pos = False
        high_water = np.nan
        bars_held = 0

        arm_th = float(params.arm_threshold)
        disarm_th = float(params.disarm_threshold)
        gap_th = float(params.gap_threshold)
        atr_mult = float(params.atr_mult)
        min_hold = int(params.min_hold_bars)

        for i in range(n):
            ac = autocorr[i]
            if not np.isnan(ac):
                if armed:
                    if ac < disarm_th:
                        armed = False
                else:
                    if ac > arm_th:
                        armed = True

            c = close[i]
            a = atr[i]

            if in_pos:
                if not np.isnan(c):
                    if np.isnan(high_water) or c > high_water:
                        high_water = c
                bars_held += 1
                exit_now = False
                if (
                    bars_held >= min_hold
                    and not np.isnan(a)
                    and not np.isnan(high_water)
                    and not np.isnan(c)
                ):
                    stop_level = high_water - atr_mult * a
                    if c < stop_level:
                        exit_now = True
                if exit_now:
                    in_pos = False
                    high_water = np.nan
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                g = gap_ret[i]
                m = ma200[i]
                if (
                    armed
                    and not np.isnan(c)
                    and not np.isnan(m)
                    and not np.isnan(g)
                    and not np.isnan(a)
                    and c > m
                    and g > gap_th
                ):
                    in_pos = True
                    high_water = c
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df = pd.DataFrame(
            {
                "signal": signal.values,
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
