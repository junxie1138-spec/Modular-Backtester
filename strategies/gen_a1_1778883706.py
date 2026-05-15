from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAutocorrParams:
    """Parameters for the gap-autocorrelation shockwave strategy."""

    gap_threshold: float = 0.0
    autocorr_window: int = 30
    autocorr_min: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapAutocorrParams]):
    """Long SPY on an up-gap inside a positively autocorrelated gap regime.

    Traffic-shockwave analogy: an overnight gap is a discrete 'shock' injected
    into the price queue. When the gap process is positively autocorrelated the
    queue is congested and shocks propagate forward across days instead of
    dissipating. The two-primitive AND requires BOTH (a) a fresh up-gap and
    (b) a persistent gap regime (rolling lag-1 gap autocorrelation above a
    positive threshold). Only when both agree is a multi-week long taken, and
    it is closed purely on a fixed bar count - no signal-based exit.
    """

    strategy_id = "gen_a1_1778883706"

    @classmethod
    def params_type(cls) -> type[GapAutocorrParams]:
        return GapAutocorrParams

    def warmup_bars(self, params: GapAutocorrParams) -> int:
        # gap consumes 1 bar; the rolling lag-1 autocorrelation consumes the
        # window plus the extra shift, plus margin for zero-variance windows.
        return int(params.autocorr_window) + 5

    def indicators(self, data: pd.DataFrame, params: GapAutocorrParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        prior_close = close.shift(1)
        gap = open_ / prior_close - 1.0

        window = max(int(params.autocorr_window), 2)
        gap_lag = gap.shift(1)
        # Vectorised rolling correlation of the gap series against its own
        # one-bar lag -> lag-1 autocorrelation of the gap process.
        gap_autocorr = gap.rolling(window).corr(gap_lag)

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_autocorr"] = gap_autocorr
        return out

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        n = len(data)
        gap = indicators["gap"].to_numpy(dtype=float)
        ac = indicators["gap_autocorr"].to_numpy(dtype=float)

        # NaN-safe: any NaN (warmup or degenerate window) fails the primitive.
        gap_ok = np.where(np.isnan(gap), False, gap > params.gap_threshold)
        ac_ok = np.where(np.isnan(ac), False, ac > params.autocorr_min)
        entry = np.asarray(gap_ok & ac_ok, dtype=bool)

        # Fixed-bar exit: each entry holds for exactly `hold` bars, then flat.
        hold = max(int(params.hold_bars), 1)
        sig = np.zeros(n, dtype=int)
        remaining = 0
        for i in range(n):
            if remaining > 0:
                sig[i] = 1
                remaining -= 1
            elif entry[i]:
                sig[i] = 1
                remaining = hold - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0

        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
