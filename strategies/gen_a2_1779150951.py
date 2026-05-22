from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 60
    gap_threshold: float = 0.003
    dd_min: float = 0.03
    dd_max: float = 0.20
    hold_bars: int = 7
    size_floor: float = 0.5
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    """Drawdown-recovery: buy when an intraday-absorbed down-gap AND a moderate
    drawdown band both agree; hold a fixed number of bars then exit."""

    strategy_id = "gen_a2_1779150951"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.peak_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        prev_close = close.shift(1)

        # Primitive 1: gap = open vs prior close.
        gap = data["open"] / prev_close - 1.0
        # An 'absorbed' down-gap: opened below prior close but closed above it.
        absorbed_gap = (gap <= -float(params.gap_threshold)) & (close > prev_close)

        # Primitive 2: drawdown depth from a rolling peak (susceptible pool).
        win = max(int(params.peak_window), 2)
        rolling_peak = close.rolling(win, min_periods=win).max()
        drawdown = close / rolling_peak - 1.0
        dd_band = (drawdown <= -float(params.dd_min)) & (drawdown >= -float(params.dd_max))

        ind = pd.DataFrame(index=data.index)
        ind["prev_close"] = prev_close
        ind["gap"] = gap
        ind["rolling_peak"] = rolling_peak
        ind["drawdown"] = drawdown
        ind["absorbed_gap"] = absorbed_gap.fillna(False)
        ind["dd_band"] = dd_band.fillna(False)
        # Two-primitive AND: both must agree on the same bar.
        ind["entry"] = (absorbed_gap & dd_band).fillna(False)
        return ind

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        entry = indicators["entry"].to_numpy(dtype=bool)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)

        hold = max(int(params.hold_bars), 1)
        dd_min = max(float(params.dd_min), 1e-6)
        floor = float(params.size_floor)
        cap = float(params.size_cap)
        if cap < floor:
            cap = floor

        raw = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        # Fixed-bar exit: once entered, hold exactly `hold` bars, no re-entry
        # while in position. Bar-indexed loop keeps the holding clock explicit.
        i = 0
        while i < n:
            if entry[i]:
                end = min(i + hold, n)
                raw[i:end] = 1
                # Epidemic flavour: deeper susceptible drawdown -> larger size.
                depth = -drawdown[i]
                if not np.isfinite(depth) or depth <= 0.0:
                    depth = dd_min
                factor = depth / dd_min
                if factor < floor:
                    factor = floor
                elif factor > cap:
                    factor = cap
                size[i:end] = factor
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=floor, upper=cap)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
