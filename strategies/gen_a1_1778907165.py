from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakMaturityParams:
    entry_streak: int = 4
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    max_hold: int = 20
    trend_ma: int = 100


class GeneratedStrategy(BaseStrategy[StreakMaturityParams]):
    strategy_id = "gen_a1_1778907165"

    @classmethod
    def params_type(cls):
        return StreakMaturityParams

    @staticmethod
    def warmup_bars(params: StreakMaturityParams) -> int:
        return int(max(params.trend_ma, params.atr_period + 1)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakMaturityParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        # consecutive up-close streak count: 1,2,3,... resetting to 0 on any non-up bar
        up = (close > prev_close).astype(int)
        reset = (up == 0).cumsum()
        streak = up.groupby(reset).cumsum().astype(float)

        # ATR via true range
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        sma = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()

        out = pd.DataFrame(index=data.index)
        out["streak"] = streak.fillna(0.0)
        out["atr"] = atr
        out["sma"] = sma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakMaturityParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        n = len(close)

        # two-bar confirmation: streak threshold satisfied on this bar AND the prior bar
        cond = streak >= float(params.entry_streak)
        confirmed = np.zeros(n, dtype=bool)
        if n > 1:
            confirmed[1:] = cond[1:] & cond[:-1]

        sig = np.zeros(n, dtype=int)
        position = 0
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                ready = (
                    bool(confirmed[i])
                    and not np.isnan(atr[i])
                    and not np.isnan(sma[i])
                    and atr[i] > 0.0
                    and close[i] > sma[i]
                )
                if ready:
                    position = 1
                    stop_level = close[i] - params.atr_stop_mult * atr[i]
                    bars_held = 0
                    sig[i] = 1
            else:
                bars_held += 1
                if close[i] <= stop_level or bars_held >= params.max_hold:
                    position = 0
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
