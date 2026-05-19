from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 5
    gap_pressure_thresh: float = 0.004
    peak_window: int = 60
    dd_thresh: float = 0.05
    atr_window: int = 14
    breakeven_pct: float = 0.03
    trail_atr_mult: float = 2.5
    init_stop_atr_mult: float = 3.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779151478"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.peak_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
        gap_pressure = gap_pct.rolling(
            int(params.gap_window), min_periods=int(params.gap_window)
        ).sum()

        pw = int(params.peak_window)
        peak = close.rolling(pw, min_periods=pw).max()
        trough = close.rolling(pw, min_periods=pw).min()
        drawdown = close / peak.replace(0.0, np.nan) - 1.0
        drawup = close / trough.replace(0.0, np.nan) - 1.0

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(
            int(params.atr_window), min_periods=int(params.atr_window)
        ).mean()

        out = pd.DataFrame(index=data.index)
        out["gap_pressure"] = gap_pressure
        out["drawdown"] = drawdown
        out["drawup"] = drawup
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)

        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        gp = indicators["gap_pressure"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        du = indicators["drawup"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        dd_thresh = float(params.dd_thresh)
        gp_thresh = float(params.gap_pressure_thresh)

        # Comparisons against NaN evaluate False -> boolean arrays are NaN-safe.
        long_cond = (dd <= -dd_thresh) & (gp > gp_thresh)
        short_cond = (du >= dd_thresh) & (gp < -gp_thresh)

        signal = np.zeros(n, dtype=np.int64)

        position = 0
        entry_price = 0.0
        stop = 0.0
        extreme = 0.0
        peaked = False
        bars_held = 0

        be_pct = float(params.breakeven_pct)
        trail_mult = float(params.trail_atr_mult)
        init_mult = float(params.init_stop_atr_mult)
        max_hold = int(params.max_hold_bars)

        for i in range(1, n):
            a = atr[i]
            if not np.isfinite(a) or a <= 0.0:
                signal[i] = position
                continue

            if position == 0:
                long_entry = bool(long_cond[i] and long_cond[i - 1])
                short_entry = bool(short_cond[i] and short_cond[i - 1])
                if long_entry:
                    position = 1
                    entry_price = close[i]
                    stop = entry_price - init_mult * a
                    extreme = high[i]
                    peaked = False
                    bars_held = 0
                    signal[i] = 1
                elif short_entry:
                    position = -1
                    entry_price = close[i]
                    stop = entry_price + init_mult * a
                    extreme = low[i]
                    peaked = False
                    bars_held = 0
                    signal[i] = -1
                else:
                    signal[i] = 0
            elif position == 1:
                bars_held += 1
                if high[i] > extreme:
                    extreme = high[i]
                if not peaked and high[i] >= entry_price * (1.0 + be_pct):
                    peaked = True
                    if entry_price > stop:
                        stop = entry_price
                if peaked:
                    trail = extreme - trail_mult * a
                    if trail > stop:
                        stop = trail
                if low[i] <= stop or bars_held >= max_hold:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:  # position == -1
                bars_held += 1
                if low[i] < extreme:
                    extreme = low[i]
                if not peaked and low[i] <= entry_price * (1.0 - be_pct):
                    peaked = True
                    if entry_price < stop:
                        stop = entry_price
                if peaked:
                    trail = extreme + trail_mult * a
                    if trail < stop:
                        stop = trail
                if high[i] >= stop or bars_held >= max_hold:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
