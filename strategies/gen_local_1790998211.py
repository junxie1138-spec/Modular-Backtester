from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorQueueParams:
    resistance_window: int = 20
    minimum_streak: int = 3
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FloorQueueParams]):
    strategy_id = "gen_local_1790998211"

    @classmethod
    def params_type(cls) -> type[FloorQueueParams]:
        return FloorQueueParams

    def warmup_bars(self, params: FloorQueueParams) -> int:
        return int(max(2, params.resistance_window) + max(1, params.minimum_streak) + 3)

    def indicators(self, data: pd.DataFrame, params: FloorQueueParams) -> pd.DataFrame:
        window = max(2, int(params.resistance_window))
        minimum = max(1, int(params.minimum_streak))
        out = pd.DataFrame(index=data.index)
        out["ceiling"] = data["high"].rolling(window, min_periods=window).max().shift(1)
        valid = data[["open", "high", "low", "close"]].notna().all(axis=1)
        out["valid"] = valid
        floor_step = (
            valid
            & valid.shift(1, fill_value=False)
            & out["ceiling"].notna()
            & out["ceiling"].eq(out["ceiling"].shift(1))
            & data["low"].gt(data["low"].shift(1))
            & data["high"].le(out["ceiling"])
            & data["close"].lt(out["ceiling"])
        ).fillna(False)
        out["floor_step"] = floor_step
        out["floor_streak"] = floor_step.astype(int).groupby((~floor_step).cumsum()).cumsum()
        out["breakout"] = (
            valid
            & out["floor_streak"].shift(1).fillna(0).ge(minimum)
            & out["ceiling"].eq(out["ceiling"].shift(1))
            & data["close"].gt(out["ceiling"])
        ).fillna(False)
        out["confirmation_boundary"] = out["ceiling"].shift(1)
        out["confirmed"] = (
            valid
            & out["breakout"].shift(1, fill_value=False)
            & data["close"].gt(out["confirmation_boundary"])
            & data["close"].ge(data["close"].shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FloorQueueParams,
    ) -> SignalFrame:
        hold = max(1, int(params.hold_bars))
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        position = np.zeros(len(data), dtype=np.int64)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar >= hold:
                    entry_bar = -1
                else:
                    position[i] = 1
            elif entries[i]:
                entry_bar = i
                position[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
