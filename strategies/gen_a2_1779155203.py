from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_lookback: int = 40
    low_band: float = 0.30
    high_band: float = 0.70
    min_streak: int = 6
    atr_period: int = 14
    atr_mult: float = 3.0
    base_size: float = 0.40
    streak_norm: float = 20.0
    max_size: float = 1.0
    max_hold: int = 12


class GeneratedStrategy(BaseStrategy[Params]):
    """Relative-position band-confinement streak release with streak-scaled sizing."""

    strategy_id = "gen_a2_1779155203"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.channel_lookback, params.atr_period)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        win = max(int(params.channel_lookback), 2)
        roll_max = close.rolling(win).max()
        roll_min = close.rolling(win).min()
        span = roll_max - roll_min
        span = span.where(span > 0.0, np.nan)
        rp = ((close - roll_min) / span).clip(0.0, 1.0)

        # Discretize relative position into low(0) / mid(1) / high(2) bands.
        band = pd.Series(1, index=close.index, dtype="int64")
        band = band.where(~(rp < float(params.low_band)), 0)
        band = band.where(~(rp > float(params.high_band)), 2)
        band = band.where(rp.notna(), 1)

        # Consecutive-streak count: length of the current run in the same band.
        grp = (band != band.shift(1)).cumsum()
        streak = band.groupby(grp).cumcount() + 1

        # ATR for the rolling-high trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(max(int(params.atr_period), 1)).mean()

        out = pd.DataFrame(index=data.index)
        out["rp"] = rp.fillna(0.5)
        out["band"] = band.astype("float64")
        out["streak"] = streak.astype("float64")
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        idx = data.index
        n = len(idx)
        close = data["close"].to_numpy(dtype="float64")
        band = indicators["band"].to_numpy(dtype="float64")
        streak = indicators["streak"].to_numpy(dtype="float64")
        atr = indicators["atr"].to_numpy(dtype="float64")

        signal = np.zeros(n, dtype="int64")
        size = np.ones(n, dtype="float64")

        warmup = GeneratedStrategy.warmup_bars(params)
        k = float(params.atr_mult)
        min_streak = int(params.min_streak)
        max_hold = int(params.max_hold)
        base_size = float(params.base_size)
        snorm = max(float(params.streak_norm), 1.0)
        max_size = float(params.max_size)

        pos = 0
        entry_size = 1.0
        hwm = 0.0
        lwm = 0.0
        bars_held = 0

        for i in range(1, n):
            if pos == 0:
                if i < warmup or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    signal[i] = 0
                    size[i] = 1.0
                    continue
                prev_band = band[i - 1]
                prev_streak = streak[i - 1]
                # Streak-scaled size: longer confinement -> larger release bet.
                scaled = base_size * (1.0 + prev_streak / snorm)
                if scaled < 0.05:
                    scaled = 0.05
                elif scaled > max_size:
                    scaled = max_size
                if prev_streak >= min_streak and prev_band == 0.0 and band[i] != 0.0:
                    pos = 1
                    hwm = close[i]
                    bars_held = 0
                    entry_size = scaled
                elif prev_streak >= min_streak and prev_band == 2.0 and band[i] != 2.0:
                    pos = -1
                    lwm = close[i]
                    bars_held = 0
                    entry_size = scaled
            else:
                bars_held += 1
                if pos == 1:
                    if close[i] > hwm:
                        hwm = close[i]
                    stop = hwm - k * atr[i]
                    if (np.isfinite(stop) and close[i] < stop) or bars_held >= max_hold:
                        pos = 0
                else:
                    if close[i] < lwm:
                        lwm = close[i]
                    stop = lwm + k * atr[i]
                    if (np.isfinite(stop) and close[i] > stop) or bars_held >= max_hold:
                        pos = 0

            signal[i] = pos
            size[i] = entry_size if pos != 0 else 1.0

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].clip(lower=0.05)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
