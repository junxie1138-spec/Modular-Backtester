from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueFillParams:
    range_window: int = 20
    roc_window: int = 5
    pos_hi: float = 0.55
    pos_lo: float = 0.40
    overflow: float = 0.92
    accel_up: float = 0.0
    accel_dn: float = 0.0


class GeneratedStrategy(BaseStrategy[QueueFillParams]):
    """Relative range-fill capacity buffer + ROC acceleration, two-primitive AND.

    Primitive A (relative position): fill level rp = where close sits inside the
    rolling high-low range, read as a queue buffer. Entry needs rp in the upper
    band (pos_hi) but below the capacity ceiling (overflow) - the buffer is
    filling, not overflowed.
    Primitive B (rate-of-change acceleration): the change in roc_window ROC must
    be positive - momentum feeding the buffer is accelerating.
    Both must agree to enter. The exit is signal-reversal: hold until the mirror
    of the entry fires (rp drained below pos_lo AND acceleration negative).
    """

    strategy_id = "gen_a1_1778895792"

    @classmethod
    def params_type(cls):
        return QueueFillParams

    def warmup_bars(self, params: QueueFillParams) -> int:
        # rp needs range_window; accel = diff of pct_change(roc_window) needs
        # roc_window + 1. Add a 2-bar buffer.
        return max(params.range_window, params.roc_window + 1) + 2

    def indicators(self, data: pd.DataFrame, params: QueueFillParams) -> pd.DataFrame:
        close = data["close"]
        high_n = data["high"].rolling(params.range_window).max()
        low_n = data["low"].rolling(params.range_window).min()

        rng = high_n - low_n
        # Guard against a flat window (high == low) producing a zero divisor.
        rng = rng.where(rng > 0.0, np.nan)
        rp = ((close - low_n) / rng).clip(0.0, 1.0)

        roc = close.pct_change(params.roc_window)
        accel = roc.diff()

        out = pd.DataFrame(index=data.index)
        out["rp"] = rp
        out["roc"] = roc
        out["accel"] = accel
        # Capacity headroom: how much room is left before overflow.
        out["headroom"] = (params.overflow - rp).clip(lower=0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QueueFillParams,
    ) -> SignalFrame:
        rp = indicators["rp"]
        accel = indicators["accel"]

        # Two-primitive AND entry: buffer filling into upper band but below
        # capacity, while ROC acceleration is positive. NaN comparisons resolve
        # to False, so warmup bars never trigger.
        entry = (
            (rp > params.pos_hi)
            & (rp < params.overflow)
            & (accel > params.accel_up)
        )
        # Mirror (signal-reversal) exit: buffer drained below the lower band
        # AND acceleration turned negative - the entry condition flipped.
        exit_cond = (rp < params.pos_lo) & (accel < params.accel_dn)

        entry_arr = entry.fillna(False).to_numpy()
        exit_arr = exit_cond.fillna(False).to_numpy()

        n = len(data)
        pos = np.zeros(n, dtype=np.int64)
        state = 0
        for i in range(n):
            if state == 0:
                if entry_arr[i]:
                    state = 1
            else:
                if exit_arr[i]:
                    state = 0
            pos[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        # Capacity-aware sizing: more headroom in the buffer => larger size.
        headroom = indicators["headroom"].fillna(0.0)
        size = (0.5 + headroom).clip(lower=0.25, upper=1.5)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
