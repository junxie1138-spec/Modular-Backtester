from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    rank_window: int = 8
    rank_threshold: int = 7
    confirm_bars: int = 3
    hold_bars: int = 7
    refractory_bars: int = 5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779877298"

    @classmethod
    def params_type(cls):
        return GenA1Params

    @classmethod
    def warmup_bars(cls, params: GenA1Params) -> int:
        return int(params.rank_window) + 1

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        w = max(2, int(params.rank_window))
        thr = float(params.rank_threshold)

        # Rank of the current close within the trailing window (1..w, min method).
        rolling_rank = close.rolling(window=w, min_periods=w).rank(method="min")

        at_top = (rolling_rank >= thr).fillna(False).astype(int)

        # Consecutive-streak length of at_top == 1, vectorised via group cumcount.
        grp = (at_top != at_top.shift()).cumsum()
        streak = at_top.groupby(grp).cumcount().add(1)
        streak = streak.where(at_top == 1, 0).astype(int)

        ind = pd.DataFrame(index=data.index)
        ind["rolling_rank"] = rolling_rank
        ind["at_top"] = at_top
        ind["top_streak"] = streak
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        streak = indicators["top_streak"].fillna(0).to_numpy(dtype=np.int64)

        confirm = max(1, int(params.confirm_bars))
        hold = max(1, int(params.hold_bars))
        refractory = max(0, int(params.refractory_bars))

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        bars_in = 0
        cooldown = 0

        for i in range(n):
            if in_pos:
                raw[i] = 1
                bars_in += 1
                if bars_in >= hold:
                    in_pos = False
                    bars_in = 0
                    cooldown = refractory
            else:
                if cooldown > 0:
                    cooldown -= 1
                    continue
                if streak[i] >= confirm:
                    in_pos = True
                    bars_in = 1
                    raw[i] = 1

        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size = pd.Series(float(max(0.0, params.size)), index=data.index)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
