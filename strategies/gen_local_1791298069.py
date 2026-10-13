from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownPhaseParams:
    peak_window: int = 63
    shallow_depth: float = 0.03
    deep_depth: float = 0.08
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownPhaseParams]):
    strategy_id = "gen_local_1791298069"

    @classmethod
    def params_type(cls):
        return DrawdownPhaseParams

    def warmup_bars(self, params: DrawdownPhaseParams) -> int:
        return max(params.peak_window, 3)

    def indicators(self, data: pd.DataFrame, params: DrawdownPhaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        peak = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1 - data["close"] / peak.where(peak.ne(0))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        change = data["close"].diff()
        up = change.gt(0)
        down = change.lt(0)
        out["alternate_down"] = down & up.shift(1, fill_value=False) & down.shift(2, fill_value=False)
        out["alternate_up"] = up & down.shift(1, fill_value=False) & up.shift(2, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownPhaseParams,
    ) -> SignalFrame:
        depth = indicators["depth"]
        enter = (((depth >= params.shallow_depth) & (depth < params.deep_depth) & indicators["alternate_down"]) | ((depth >= params.deep_depth) & indicators["alternate_up"])).fillna(False).to_numpy(dtype=bool)
        enter[:self.warmup_bars(params)] = False
        state = [0] * len(data)
        remaining = 0
        for i, eligible in enumerate(enter):
            if remaining:
                remaining -= 1
                state[i] = int(remaining > 0)
            elif eligible:
                remaining = params.hold_bars
                state[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = state
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
