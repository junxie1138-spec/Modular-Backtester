from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    dd_window: int = 60
    rank_window: int = 120
    deep_pct: float = 0.25
    capacity: int = 5
    profit_target: float = 0.03
    time_stop: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779153480"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.dd_window, params.rank_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        dd_w = int(params.dd_window)
        rank_w = int(params.rank_window)

        ind = pd.DataFrame(index=data.index)

        roll_max = close.rolling(dd_w, min_periods=dd_w).max()
        drawdown = close / roll_max - 1.0
        ind["drawdown"] = drawdown

        # A bar prints a fresh drawdown low when close ties/undercuts the
        # rolling-window minimum close.
        roll_min = close.rolling(dd_w, min_periods=dd_w).min()
        new_low = (close <= roll_min) & roll_min.notna()
        ind["new_low"] = new_low.astype(int)

        # Capacity-limited queue: every no-new-low bar adds a token, every
        # fresh low drains the queue to zero. streak = tokens in the queue.
        group = new_low.cumsum()
        no_new_low = (~new_low).astype(int)
        streak = no_new_low.groupby(group).cumsum()
        ind["streak"] = streak.astype(float)

        # Twist: deep-drawdown test is a percentile rank of the drawdown
        # series itself, not a fixed -X% level. Deepest drawdowns rank low.
        dd_rank = drawdown.rolling(rank_w, min_periods=rank_w).rank(pct=True)
        ind["dd_rank"] = dd_rank

        deep = (dd_rank <= float(params.deep_pct)).fillna(False)
        overflow = streak >= int(params.capacity)
        green = (close > close.shift(1)).fillna(False)
        entry = deep & overflow & green
        ind["entry"] = entry.fillna(False).astype(int)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_cond = indicators["entry"].to_numpy(dtype=bool)

        pt = float(params.profit_target)
        ts = int(params.time_stop)

        raw = np.zeros(n, dtype=int)
        in_pos = False
        entry_idx = 0
        entry_price = 0.0

        for i in range(n):
            if not in_pos:
                if entry_cond[i]:
                    in_pos = True
                    entry_idx = i
                    entry_price = close[i]
                    raw[i] = 1
            else:
                bars_held = i - entry_idx
                gain = close[i] / entry_price - 1.0 if entry_price > 0 else 0.0
                if gain >= pt or bars_held >= ts:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
