from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    # Window for the drawdown reference high and the depth-elevation median.
    lookback: int = 60
    # Fixed-bar holding horizon (~1-2 weeks of daily bars).
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779155540"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling max / rolling median need `lookback`; depth.diff() needs +1.
        return int(params.lookback) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        lb = max(int(params.lookback), 2)
        close = data["close"].astype(float)

        roll_max = close.rolling(lb, min_periods=lb).max()
        # Drawdown depth: positive fraction below the trailing high.
        depth = (roll_max - close) / roll_max.replace(0.0, np.nan)
        depth = depth.clip(lower=0.0)

        depth_med = depth.rolling(lb, min_periods=lb).median()
        depth_diff = depth.diff()

        out = pd.DataFrame(index=data.index)
        out["depth"] = depth
        out["depth_med"] = depth_med
        out["depth_diff"] = depth_diff
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        hold = max(int(params.hold_bars), 1)

        depth = indicators["depth"]
        depth_med = indicators["depth_med"]
        depth_diff = indicators["depth_diff"]

        # Predator-prey turning point: drawdown depth was growing yesterday and
        # is no longer growing today (the predator population has peaked) ...
        peaked = (depth_diff.shift(1) > 0.0) & (depth_diff <= 0.0)
        # ... and the drawdown is genuinely elevated, not cosmetic noise.
        elevated = (depth > depth_med) & (depth > 0.0)

        entry = (peaked & elevated).fillna(False).to_numpy()

        # Size scales with how deep the drawdown is relative to its norm.
        ratio = depth / depth_med.replace(0.0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        size_series = ratio.clip(lower=1.0, upper=3.0).fillna(1.0)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_i = -1
        for i in range(n):
            if in_pos:
                if i - entry_i >= hold:
                    in_pos = False
                else:
                    raw[i] = 1
            if not in_pos and entry[i]:
                in_pos = True
                entry_i = i
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size_series.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
