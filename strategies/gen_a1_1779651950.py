from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    # Bucket capacity in basis points. Returns are summed into a clamped
    # accumulator that saturates at +/- capacity_bp.
    capacity_bp: float = 250.0
    # Fraction of capacity required to trigger entry (and, symmetrically,
    # exit on the opposite rail). 1.0 means full saturation is required.
    entry_pct: float = 0.95


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779651950"

    @classmethod
    def params_type(cls):
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        # The accumulator is a state variable with no fixed lookback, but a
        # short warmup lets the bucket evolve away from its zero initial value
        # before any decisions matter.
        return 30

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()
        # Convert to basis points and scrub any non-finite values from data gaps.
        ret_bp = (ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)) * 10000.0

        out = pd.DataFrame(index=data.index)
        out["ret_bp"] = ret_bp
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        ret_bp = indicators["ret_bp"].to_numpy(dtype=np.float64)
        n = ret_bp.shape[0]

        cap = float(params.capacity_bp)
        if cap <= 0.0:
            cap = 1.0
        pct = float(params.entry_pct)
        if pct <= 0.0:
            pct = 1.0
        if pct > 1.0:
            pct = 1.0
        trigger = cap * pct

        bucket = np.zeros(n, dtype=np.float64)
        raw_signal = np.zeros(n, dtype=np.int64)

        # Path-dependent: token bucket with hard saturation rails plus a
        # latched position state. Entry condition ("bucket at upper rail")
        # only flips when its mirror image fires on the lower rail - this is
        # the signal-reversal exit.
        position = 0
        b = 0.0
        for i in range(n):
            r = ret_bp[i]
            if not np.isfinite(r):
                r = 0.0
            b = b + r
            if b > cap:
                b = cap
            elif b < -cap:
                b = -cap
            bucket[i] = b

            if position == 0:
                if b >= trigger:
                    position = 1
            else:
                # Signal-reversal exit: the entry condition flips only when
                # the bucket saturates against the opposite rail.
                if b <= -trigger:
                    position = 0

            raw_signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal_raw"] = raw_signal
        # Mandatory one-bar shift: decision on bar N's close fills on bar N+1.
        df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
