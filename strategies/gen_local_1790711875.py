from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_streak: int = 3
    min_displacement: float = 0.02
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790711875"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return 6

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        direction = np.sign(data["close"].pct_change()).fillna(0).astype(int)
        group = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(group).cumcount().add(1).where(direction.ne(0), 0)
        displacement = data["close"].pct_change(5)
        return pd.DataFrame({"direction": direction, "streak": streak, "displacement": displacement}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        qualified = (
            (indicators["streak"] >= params.min_streak)
            & (indicators["displacement"].abs() >= params.min_displacement)
            & (np.sign(indicators["displacement"]) == indicators["direction"])
        )
        candidate = indicators["direction"].where(qualified, 0).fillna(0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, proposed in enumerate(candidate.to_numpy()):
            if position and proposed == -position:
                position = 0
            elif not position and proposed:
                position = int(proposed)
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": params.size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
