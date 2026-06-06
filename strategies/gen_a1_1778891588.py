from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTrendParams:
    min_gap: float = 0.0015
    coh_window: int = 20
    coh_threshold: float = 0.2
    cap_streak: int = 4
    profit_target: float = 0.015
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapTrendParams]):
    strategy_id = "gen_a1_1778891588"

    @classmethod
    def params_type(cls):
        return GapTrendParams

    @staticmethod
    def warmup_bars(params: GapTrendParams) -> int:
        return int(params.coh_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTrendParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        prior_close = close.shift(1)
        gap = (open_ - prior_close) / prior_close

        up = gap > params.min_gap
        down = gap < -params.min_gap

        L = int(params.coh_window)
        up_count = up.astype(float).rolling(L).sum()
        down_count = down.astype(float).rolling(L).sum()
        coherence = (up_count - down_count) / float(L)

        # consecutive up-gap streak: cumulative count of up bars since last non-up
        grp = (~up).cumsum()
        streak = up.astype(int).groupby(grp).cumsum()

        ind = pd.DataFrame(index=data.index)
        ind["gap"] = gap.fillna(0.0)
        ind["gap_up"] = up.astype(float)
        ind["coherence"] = coherence.fillna(0.0)
        ind["streak"] = streak.astype(float).fillna(0.0)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTrendParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        gap_up = indicators["gap_up"].to_numpy(dtype=float) > 0.5
        coherence = indicators["coherence"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        n = len(close)

        # Two-primitive AND: fresh up-gap (primitive 1) must agree with
        # coherent upward gap-sign trend strength (primitive 2).
        # Capacity gate: skip exhausted up-gap runs.
        entry = (
            gap_up
            & (coherence >= params.coh_threshold)
            & (streak <= float(params.cap_streak))
        )

        pt = float(params.profit_target)
        max_hold = int(params.max_hold)

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                cur = close[i]
                hit_pt = cur >= entry_price * (1.0 + pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    raw[i] = 0
                    in_pos = False
                    bars_held = 0
                    entry_price = 0.0
                else:
                    raw[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
