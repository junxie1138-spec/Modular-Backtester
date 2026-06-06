from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeasonalityParams:
    sma_window: int = 200
    gap_window: int = 20
    gap_pctile: float = 0.80
    hold_bars: int = 8
    require_uptrend: bool = True


class GeneratedStrategy(BaseStrategy[GapSeasonalityParams]):
    strategy_id = "gen_a1_1778895215"

    @classmethod
    def params_type(cls):
        return GapSeasonalityParams

    @staticmethod
    def warmup_bars(params: GapSeasonalityParams) -> int:
        # gap_window same-weekday occurrences span ~gap_window*5 calendar bars;
        # +1 for the prior-close shift used to build the gap series.
        return max(int(params.sma_window), int(params.gap_window) * 5 + 10) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapSeasonalityParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0

        sma = close.rolling(
            int(params.sma_window), min_periods=int(params.sma_window)
        ).mean()

        weekday = pd.Series(data.index.dayofweek, index=data.index)

        # Twist: the entry threshold is a PERCENTILE of each weekday's own
        # trailing gap distribution, not a fixed gap level. The percentile is
        # computed on the same-weekday subseries shifted by one occurrence so
        # today's gap is never inside its own reference window.
        w = int(params.gap_window)
        p = float(params.gap_pctile)
        gap_thr = pd.Series(np.nan, index=data.index, dtype=float)
        for wd in range(5):
            mask = (weekday == wd)
            if not bool(mask.any()):
                continue
            g = gap[mask]
            thr = g.shift(1).rolling(w, min_periods=w).quantile(p)
            gap_thr.loc[mask] = thr

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["sma"] = sma
        out["gap_thr"] = gap_thr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSeasonalityParams,
    ) -> SignalFrame:
        gap = indicators["gap"]
        sma = indicators["sma"]
        gap_thr = indicators["gap_thr"]
        close = data["close"]

        # Fire when the overnight gap is positive AND exceeds the high
        # percentile of its weekday's trailing gap climatology. NaN-safe:
        # comparisons against NaN thresholds evaluate False.
        fire = (gap > gap_thr) & (gap > 0.0)
        if bool(params.require_uptrend):
            fire = fire & (close > sma)
        fire_arr = fire.fillna(False).to_numpy()

        n = len(data)
        sig = np.zeros(n, dtype=int)
        hold_n = max(1, int(params.hold_bars))

        # Fixed-bar exit: once long, hold exactly hold_n bars then flatten.
        # No signal-based exit; a fresh fire may re-enter on the next bar.
        in_pos = False
        held = 0
        for i in range(n):
            if in_pos:
                sig[i] = 1
                held += 1
                if held >= hold_n:
                    in_pos = False
                    held = 0
            elif fire_arr[i]:
                in_pos = True
                sig[i] = 1
                held = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
