from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakDominanceParams:
    min_streak: int = 2
    dominance_margin: int = 1


class GeneratedStrategy(BaseStrategy[StreakDominanceParams]):
    strategy_id = "gen_local-w2_1791001308"

    @classmethod
    def params_type(cls):
        return StreakDominanceParams

    def warmup_bars(self, params: StreakDominanceParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: StreakDominanceParams) -> pd.DataFrame:
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        direction = np.sign(close.diff()).fillna(0).astype(int)
        history = pd.DataFrame(
            {lag: direction.shift(lag).fillna(0) for lag in range(8)},
            index=data.index,
        )
        same = history.eq(direction, axis=0) & direction.ne(0).to_numpy()[:, None]
        current = same.astype(int).cumprod(axis=1).sum(axis=1)
        previous = pd.Series(0, index=data.index, dtype=int)
        alive = pd.Series(True, index=data.index)
        for lag in range(8):
            in_previous = current.le(lag)
            alive &= ~in_previous | history[lag].eq(-direction)
            previous += (in_previous & alive & direction.ne(0)).astype(int)
        ready = close.notna().astype(int).rolling(9, min_periods=9).sum().eq(9)
        return pd.DataFrame(
            {
                "direction": direction,
                "current_streak": current,
                "previous_streak": previous,
                "ready": ready,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakDominanceParams,
    ) -> SignalFrame:
        direction = indicators["direction"]
        dominates = indicators["current_streak"].ge(
            indicators["previous_streak"] + params.dominance_margin
        )
        eligible = indicators["ready"] & indicators["current_streak"].ge(params.min_streak)
        candidate = direction.where(dominates, -direction).where(eligible, 0).fillna(0).astype(int)
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, entry in enumerate(candidate.to_numpy()):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # An opposing entry exits; a later qualifying bar may re-enter.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
