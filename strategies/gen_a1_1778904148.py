from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownTideParams:
    window: int = 20
    band: float = 0.25


class GeneratedStrategy(BaseStrategy[DrawdownTideParams]):
    """Trend-strength via drawdown depth relative to its characteristic tide level.

    The drawdown of close below a rolling high-water mark behaves like a tide
    level: it has a characteristic mean depth over any window. When the current
    drawdown sits above that mean depth (water above mean tide) the uptrend is
    robust and we hold long; when it sinks below the mean depth the trend is
    eroding and we hold short. A deadband proportional to the drawdown's own
    dispersion provides hysteresis, and the position is held until the entry
    condition flips to the opposite side (signal-reversal exit).
    """

    strategy_id = "gen_a1_1778904148"

    @classmethod
    def params_type(cls):
        return DrawdownTideParams

    def warmup_bars(self, params):
        # dd needs `window` bars; mean/std of dd need another `window` valid
        # values, so the gap series is only valid after ~2*window bars.
        return int(params.window) * 2 + 2

    def indicators(self, data, params):
        close = data["close"].astype(float)
        w = max(2, int(params.window))

        hwm = close.rolling(w, min_periods=w).max()
        dd = close / hwm - 1.0  # <= 0, drawdown from rolling high-water mark

        mean_dd = dd.rolling(w, min_periods=w).mean()  # characteristic tide level
        std_dd = dd.rolling(w, min_periods=w).std(ddof=0)

        gap = dd - mean_dd  # > 0: above mean tide (strong); < 0: below (weak)
        threshold = float(params.band) * std_dd

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["gap"] = gap
        out["threshold"] = threshold
        return out

    def generate_signals(self, data, indicators, ctx, params):
        gap = indicators["gap"].to_numpy(dtype=float)
        thr = indicators["threshold"].to_numpy(dtype=float)
        n = len(data.index)

        raw = np.zeros(n, dtype=np.int64)
        pos = 0  # current target position: -1 / 0 / +1

        # Path-dependent state machine: flip on band crossings, otherwise hold.
        # The position is only ever closed by flipping to the opposite side,
        # which is the mandated signal-reversal exit.
        for i in range(n):
            g = gap[i]
            t = thr[i]
            if np.isfinite(g) and np.isfinite(t) and t > 0.0:
                if pos <= 0 and g > t:
                    pos = 1
                elif pos >= 0 and g < -t:
                    pos = -1
            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        # Decide on bar N's close, fill on bar N+1.
        df["signal"] = (
            pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
