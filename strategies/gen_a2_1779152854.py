from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ac_window: int = 40
    arm_thr: float = 0.10
    disarm_thr: float = -0.05
    gap_threshold: float = 0.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779152854"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.ac_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0
        gap = gap.replace([np.inf, -np.inf], np.nan)
        w = max(int(params.ac_window), 5)
        # Rolling lag-1 autocorrelation of the overnight gap series.
        ac = gap.rolling(w, min_periods=w).corr(gap.shift(1))
        ac = ac.replace([np.inf, -np.inf], np.nan)
        up_gap = (gap > float(params.gap_threshold)).astype(float)
        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["ac"] = ac
        out["up_gap"] = up_gap
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        ac = indicators["ac"].to_numpy(dtype=float)
        up = indicators["up_gap"].to_numpy(dtype=float)
        raw = np.zeros(n, dtype=int)

        hold = max(int(params.hold_bars), 1)
        arm = float(params.arm_thr)
        disarm = float(params.disarm_thr)
        if disarm > arm:
            disarm = arm
        start = max(int(params.ac_window) + 2, 2)

        armed = False
        in_pos = False
        bars_held = 0

        for i in range(start, n):
            a = ac[i]
            # Hysteresis-latched gap-autocorrelation regime.
            if not np.isnan(a):
                if armed and a < disarm:
                    armed = False
                elif (not armed) and a > arm:
                    armed = True

            # Fixed-bar exit: close exactly `hold` bars after entry.
            if in_pos and bars_held >= hold:
                in_pos = False

            if in_pos:
                raw[i] = 1
                bars_held += 1
            else:
                # Two-bar confirmation: up-gap on this bar and the prior bar.
                if armed and up[i] > 0.5 and up[i - 1] > 0.5:
                    in_pos = True
                    bars_held = 1
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
