from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_streak: int = 2
    predator_window: int = 20
    predator_ceiling: float = 0.55
    atr_period: int = 14
    atr_mult: float = 2.0
    max_streak_cap: int = 6


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778898857"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.predator_window + 2, params.atr_period) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        up = (close > close.shift(1)).astype(int)
        down = (close < close.shift(1)).astype(int)

        # Consecutive up-close streak count (vectorized: groupby cumsum resets at each down bar)
        groups = (1 - up).cumsum()
        ind["streak"] = up.groupby(groups).cumsum().astype(float)

        # Predator density: rolling fraction of down-closes over long window
        ind["predator_density"] = down.rolling(params.predator_window).mean()

        # Predator declining: density 2 bars ago was higher than now (NaN comparison -> False -> 0.0)
        ind["predator_declining"] = (ind["predator_density"].diff(2) < 0).astype(float)

        # ATR for trailing stop
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        streak = indicators["streak"].values
        predator_density = indicators["predator_density"].values
        predator_declining = indicators["predator_declining"].values
        atr = indicators["atr"].values
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, 0.30)

        in_trade = False
        high_water = 0.0
        stop_level = 0.0
        entry_size = 0.30

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(predator_density[i]):
                continue

            if in_trade:
                # Ratchet high-water mark and trailing stop upward only
                if close[i] > high_water:
                    high_water = close[i]
                    stop_level = high_water - params.atr_mult * atr[i]

                if close[i] <= stop_level:
                    raw_signal[i] = 0
                    in_trade = False
                else:
                    raw_signal[i] = 1
                    raw_size[i] = entry_size
            else:
                # Entry: streak meets threshold, predators below ceiling and declining
                if (
                    streak[i] >= params.min_streak
                    and predator_density[i] <= params.predator_ceiling
                    and predator_declining[i] == 1.0
                ):
                    # Signal-scaled sizing: longer streak -> larger position (prey-burst confidence)
                    frac = min(streak[i] / max(params.max_streak_cap, 1), 1.0)
                    entry_size = float(0.30 + 0.65 * frac)
                    raw_signal[i] = 1
                    raw_size[i] = entry_size
                    in_trade = True
                    high_water = close[i]
                    stop_level = high_water - params.atr_mult * atr[i]

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(raw_size, index=data.index).shift(1).fillna(0.30).clip(lower=0.30)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
