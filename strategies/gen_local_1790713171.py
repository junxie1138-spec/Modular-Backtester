from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_bars: int = 3
    momentum_bars: int = 5
    atr_bars: int = 14
    breakeven_pct: float = 0.02
    atr_trail: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713171"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.momentum_bars + 1, params.atr_bars)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        up = close.gt(close.shift(1))
        down = close.lt(close.shift(1))
        up_streak = up.astype(int).groupby((~up).cumsum()).cumsum()
        down_streak = down.astype(int).groupby((~down).cumsum()).cumsum()
        momentum = close.pct_change(params.momentum_bars)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return pd.DataFrame({"up_streak": up_streak, "down_streak": down_streak, "momentum": momentum, "atr": atr}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        pos = 0
        entry = 0.0
        best = 0.0
        stop = np.nan
        streak_up = indicators["up_streak"].fillna(0).to_numpy()
        streak_down = indicators["down_streak"].fillna(0).to_numpy()
        momentum = indicators["momentum"].to_numpy()
        atr = indicators["atr"].to_numpy()
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        close = data["close"].to_numpy()

        for i in range(n):
            exited = False
            if pos == 1:
                if high[i] >= entry * (1.0 + params.breakeven_pct):
                    best = max(best, high[i])
                    candidate = max(entry, best - params.atr_trail * atr[i]) if np.isfinite(atr[i]) else entry
                    stop = max(stop, candidate) if np.isfinite(stop) else candidate
                if np.isfinite(stop) and low[i] <= stop:
                    pos = 0
                    stop = np.nan
                    exited = True
            elif pos == -1:
                if low[i] <= entry * (1.0 - params.breakeven_pct):
                    best = min(best, low[i])
                    candidate = min(entry, best + params.atr_trail * atr[i]) if np.isfinite(atr[i]) else entry
                    stop = min(stop, candidate) if np.isfinite(stop) else candidate
                if np.isfinite(stop) and high[i] >= stop:
                    pos = 0
                    stop = np.nan
                    exited = True

            if pos == 0 and not exited and np.isfinite(momentum[i]) and np.isfinite(atr[i]):
                if streak_up[i] >= params.streak_bars and momentum[i] > 0:
                    pos = 1
                    entry = close[i]
                    best = high[i]
                    stop = np.nan
                elif streak_down[i] >= params.streak_bars and momentum[i] < 0:
                    pos = -1
                    entry = close[i]
                    best = low[i]
                    stop = np.nan
            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = sizes
        return SignalFrame(data=df, signal_column="signal", size_column="size")
