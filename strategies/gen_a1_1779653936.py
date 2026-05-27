from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HysteresisGapParams:
    upper_gap_pct: float = 0.003
    lower_gap_pct: float = -0.001
    smooth_bars: int = 1
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[HysteresisGapParams]):
    strategy_id = "gen_a1_1779653936"

    @classmethod
    def params_type(cls):
        return HysteresisGapParams

    @classmethod
    def warmup_bars(cls, params: HysteresisGapParams) -> int:
        smooth_n = max(1, int(params.smooth_bars))
        return max(smooth_n + 1, 2)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: HysteresisGapParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        prev_close_safe = prev_close.where(prev_close.abs() > 1e-12, np.nan)
        gap = (data["open"] - prev_close_safe) / prev_close_safe
        smooth_n = max(1, int(params.smooth_bars))
        gap_smooth = gap.rolling(smooth_n, min_periods=1).mean()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_smooth"] = gap_smooth
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HysteresisGapParams,
    ) -> SignalFrame:
        gap_smooth = indicators["gap_smooth"].to_numpy(dtype=float)

        upper = float(params.upper_gap_pct)
        lower = float(params.lower_gap_pct)
        if lower > upper:
            lower, upper = upper, lower

        n = len(data)
        state = np.zeros(n, dtype=np.int64)
        cur = 0
        for i in range(n):
            g = gap_smooth[i]
            if not np.isfinite(g):
                state[i] = cur
                continue
            if cur == 0:
                if g >= upper:
                    cur = 1
            else:
                if g <= lower:
                    cur = 0
            state[i] = cur

        df = pd.DataFrame(index=data.index)
        raw_signal = pd.Series(state, index=data.index, dtype="int64")
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)

        size_val = float(params.base_size)
        if not np.isfinite(size_val) or size_val <= 0.0:
            size_val = 1.0
        df["size"] = size_val

        return SignalFrame(data=df, signal_column="signal", size_column="size")
