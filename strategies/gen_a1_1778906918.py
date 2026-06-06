from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_window: int = 20
    smooth_window: int = 5
    accel_window: int = 5
    hold_bars: int = 18
    refractory_bars: int = 5
    min_trend: float = 0.004


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778906918"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return (
            int(params.roc_window)
            + int(params.smooth_window)
            + int(params.accel_window)
            + 2
        )

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        roc_window = max(1, int(params.roc_window))
        smooth_window = max(1, int(params.smooth_window))
        accel_window = max(1, int(params.accel_window))

        # First derivative: rate of change of price over roc_window bars.
        roc = close.pct_change(roc_window)
        # Smoothed trend strength (primitive A source).
        roc_smooth = roc.rolling(smooth_window, min_periods=smooth_window).mean()
        # Second derivative: acceleration of the smoothed ROC (primitive B source).
        accel = roc_smooth - roc_smooth.shift(accel_window)

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["roc_smooth"] = roc_smooth
        out["accel"] = accel
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        roc_smooth = indicators["roc_smooth"]
        accel = indicators["accel"]
        min_trend = float(params.min_trend)

        # Two-primitive AND: trend sign (A) must agree with acceleration sign (B).
        raw_long = (roc_smooth > min_trend) & (accel > 0.0)
        raw_short = (roc_smooth < -min_trend) & (accel < 0.0)
        raw_long = raw_long.fillna(False).to_numpy()
        raw_short = raw_short.fillna(False).to_numpy()

        n = len(data)
        sig = np.zeros(n, dtype=np.int64)
        hold = max(1, int(params.hold_bars))
        refr = max(0, int(params.refractory_bars))

        pos = 0
        entry_idx = -1
        cooldown = 0
        for i in range(n):
            if pos != 0:
                # Fixed-bar exit: flatten exactly hold bars after entry.
                if i - entry_idx >= hold:
                    pos = 0
                    cooldown = refr
                    sig[i] = 0
                else:
                    sig[i] = pos
            if pos == 0:
                if cooldown > 0:
                    # Refractory lockout after a closed trade.
                    cooldown -= 1
                elif raw_long[i]:
                    pos = 1
                    entry_idx = i
                    sig[i] = 1
                elif raw_short[i]:
                    pos = -1
                    entry_idx = i
                    sig[i] = -1

        out = pd.DataFrame(index=data.index)
        out["signal"] = sig
        out["size"] = 1.0
        out["signal"] = out["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
