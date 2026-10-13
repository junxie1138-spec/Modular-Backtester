from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_trigger: int = 4
    min_hold_bars: int = 15
    saturation_streak: int = 7
    size_floor: float = 0.35
    size_ceiling: float = 0.95


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1791226716"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    @classmethod
    def warmup_bars(cls, params: StreakParams) -> int:
        return 1

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        direction = np.sign(data["close"].diff().fillna(0)).astype(int)
        groups = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(groups).cumcount().add(1).mul(direction)
        return pd.DataFrame({"streak": streak.fillna(0).astype(int)}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        signals: list[int] = []
        sizes: list[float] = []
        position = 0
        entry_bar = -1
        held_size = max(0.01, params.size_floor)
        span = max(1, params.saturation_streak - params.streak_trigger)

        for bar, streak in enumerate(indicators["streak"].fillna(0).astype(int)):
            candidate = 1 if streak <= -params.streak_trigger else -1 if streak >= params.streak_trigger else 0
            strength = min(1.0, max(0.0, (abs(streak) - params.streak_trigger) / span))
            candidate_size = max(0.01, params.size_floor + (params.size_ceiling - params.size_floor) * strength)

            if position == 0 and candidate:
                position = candidate
                entry_bar = bar
                held_size = candidate_size
            elif candidate == -position and position and bar - entry_bar >= params.min_hold_bars:
                position = 0
            elif candidate == position and position:
                held_size = candidate_size

            signals.append(position)
            sizes.append(held_size)

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(max(0.01, params.size_floor)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
