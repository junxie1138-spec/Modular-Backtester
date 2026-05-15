from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDrawdownParams:
    peak_lookback: int = 126
    dd_arm: float = 0.06
    dd_disarm: float = 0.02
    gap_threshold: float = 0.002
    hold_bars: int = 8
    size_base: float = 0.40
    size_dd_mult: float = 6.0
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[GapDrawdownParams]):
    strategy_id = "gen_a1_1778884834"

    @classmethod
    def params_type(cls):
        return GapDrawdownParams

    @classmethod
    def warmup_bars(cls, params: GapDrawdownParams) -> int:
        # rolling peak window + 1 for the close.shift(1) used by the gap series
        return int(params.peak_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: GapDrawdownParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        lookback = max(int(params.peak_lookback), 1)
        # rolling peak; min_periods=1 keeps early bars NaN-safe
        peak = close.rolling(lookback, min_periods=1).max()
        # drawdown depth as a positive number: 0 at the peak, larger when deeper
        dd_depth = (1.0 - close / peak).clip(lower=0.0)

        # overnight gap: today's open relative to yesterday's close
        gap = (open_ / close.shift(1)) - 1.0

        ind = pd.DataFrame(index=data.index)
        ind["dd_depth"] = dd_depth.fillna(0.0)
        ind["gap"] = gap.fillna(0.0)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDrawdownParams,
    ) -> SignalFrame:
        depth = indicators["dd_depth"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        n = len(data.index)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, float(params.size_base), dtype=float)

        hold = max(int(params.hold_bars), 1)
        arm = float(params.dd_arm)
        disarm = float(params.dd_disarm)
        # guard the hysteresis band: arm threshold must sit above disarm threshold
        if arm <= disarm:
            arm = disarm + 1e-6

        armed = False
        i = 0
        while i < n:
            d = depth[i] if np.isfinite(depth[i]) else 0.0
            g = gap[i] if np.isfinite(gap[i]) else 0.0

            # hysteresis state machine on drawdown depth
            if not armed and d >= arm:
                armed = True
            elif armed and d <= disarm:
                armed = False

            # entry: first up-gap while armed
            if armed and g >= float(params.gap_threshold):
                sz = float(params.size_base) + float(params.size_dd_mult) * d
                sz = float(min(max(sz, 0.05), float(params.size_max)))
                end = min(i + hold, n)
                signal[i:end] = 1
                size[i:end] = sz
                armed = False  # disarm after committing the trade
                i = end  # fixed-bar exit: no overlapping positions
                continue
            i += 1

        df = pd.DataFrame(index=data.index)
        # mandatory one-bar shift: decision on bar N close, fill on bar N+1
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = (
            pd.Series(size, index=data.index)
            .shift(1)
            .fillna(float(params.size_base))
            .clip(lower=0.05)
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
