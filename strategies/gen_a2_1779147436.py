from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 40
    elastic_limit: float = 0.06


class GeneratedStrategy(BaseStrategy[Params]):
    """Drawdown-recovery long: buy shallow within-elastic-limit dips on the
    first positive close-to-close return; reject deep (plastic) drawdowns;
    hold a fixed number of bars then exit unconditionally."""

    strategy_id = "gen_a2_1779147436"

    # Fixed-bar exit horizon (~1.5 trading weeks). Not tunable on purpose:
    # the hard twist caps tunable params at 2 (peak_window, elastic_limit).
    HOLD_BARS = 7

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling peak of length peak_window, plus 1 for pct_change.
        return int(params.peak_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        win = max(int(params.peak_window), 1)

        ret = close.pct_change()
        peak = close.rolling(win, min_periods=win).max()
        dd = close / peak - 1.0  # <= 0, NaN during warmup

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["dd"] = dd
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        idx = data.index
        n = len(idx)

        ret = indicators["ret"].to_numpy()
        dd = indicators["dd"].to_numpy()
        limit = abs(float(params.elastic_limit))
        hold = int(GeneratedStrategy.HOLD_BARS)

        # Entry trigger: drawdown is a genuine dip but still inside the
        # elastic limit, AND the close-to-close return has just turned
        # positive (the spring beginning to snap back).
        entry = np.zeros(n, dtype=bool)
        for i in range(n):
            d = dd[i]
            r = ret[i]
            if np.isfinite(d) and np.isfinite(r):
                entry[i] = (d < 0.0) and (d >= -limit) and (r > 0.0)

        # Fixed-bar exit: once long, hold exactly HOLD_BARS bars, then flat.
        # No new entries while a position is open.
        raw = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if entry[i]:
                end = min(i + hold, n)
                raw[i:end] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(raw, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
