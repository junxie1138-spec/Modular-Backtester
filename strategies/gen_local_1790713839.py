from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    streak_length: int = 2
    noise_window: int = 5
    min_efficiency: float = 0.5


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1790713839"

    @classmethod
    def params_type(cls):
        return MomentumParams

    @classmethod
    def warmup_bars(cls, params: MomentumParams) -> int:
        return params.noise_window + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        close = data["close"]
        change = close.diff()
        direction = np.sign(change).fillna(0).astype(int)
        run_id = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(run_id).cumcount() + 1
        streak = streak.where(direction.ne(0), 0)
        path = change.abs().rolling(params.noise_window, min_periods=params.noise_window).sum()
        efficiency = (close - close.shift(params.noise_window)).abs().div(path.replace(0, np.nan))
        return pd.DataFrame({"direction": direction, "streak": streak, "efficiency": efficiency}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        efficient = indicators["efficiency"].ge(params.min_efficiency).fillna(False)
        long_base = indicators["direction"].eq(1) & indicators["streak"].ge(params.streak_length) & efficient
        short_base = indicators["direction"].eq(-1) & indicators["streak"].ge(params.streak_length) & efficient
        long_entry = (long_base & long_base.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_base & short_base.shift(1, fill_value=False)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1 and short_entry[i]:
                position = 0
            elif position == -1 and long_entry[i]:
                position = 0
            elif position == 0:
                if long_entry[i]:
                    position = 1
                elif short_entry[i]:
                    position = -1
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
