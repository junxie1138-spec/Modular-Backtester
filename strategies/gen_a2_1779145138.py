from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReversalParams:
    ret_lookback: int = 20
    ret_z: float = 1.5
    gap_thresh: float = 0.003
    hold_bars: int = 4
    refractory_bars: int = 3


class GeneratedStrategy(BaseStrategy[GapReversalParams]):
    strategy_id = "gen_a2_1779145138"

    @classmethod
    def params_type(cls) -> type[GapReversalParams]:
        return GapReversalParams

    @staticmethod
    def warmup_bars(params: GapReversalParams) -> int:
        return int(params.ret_lookback) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapReversalParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        # Primitive A: close-to-close return spike (z-scored against rolling vol).
        cc_ret = close.pct_change()
        ret_std = cc_ret.rolling(params.ret_lookback).std()
        down_spike = cc_ret < (-params.ret_z * ret_std)
        down_spike = down_spike.fillna(False)

        # Primitive B: the session opened with a down gap vs the prior close.
        gap = open_ / close.shift(1) - 1.0
        gap_down = gap < (-params.gap_thresh)
        gap_down = gap_down.fillna(False)

        # Hard twist: two-primitive AND - both must agree to flag an entry.
        entry_flag = (down_spike & gap_down).astype(float)

        out = pd.DataFrame(index=data.index)
        out["cc_ret"] = cc_ret.fillna(0.0)
        out["ret_std"] = ret_std.fillna(0.0)
        out["gap"] = gap.fillna(0.0)
        out["down_spike"] = down_spike.astype(float)
        out["gap_down"] = gap_down.astype(float)
        out["entry_flag"] = entry_flag
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReversalParams,
    ) -> SignalFrame:
        n = len(data)
        entry_flag = indicators["entry_flag"].to_numpy() > 0.5

        hold_bars = max(1, int(params.hold_bars))
        refractory_bars = max(0, int(params.refractory_bars))

        raw = np.zeros(n, dtype=np.int64)
        hold_left = 0
        refrac_left = 0

        # Fixed-bar exit: once entered, stay long exactly hold_bars bars, then
        # force the signal to 0. No signal-based exit. A refractory cooldown
        # after each trade prevents clustered capitulation days from stacking.
        for i in range(n):
            if hold_left > 0:
                raw[i] = 1
                hold_left -= 1
                if hold_left == 0:
                    refrac_left = refractory_bars
                continue
            if refrac_left > 0:
                refrac_left -= 1
                continue
            if entry_flag[i]:
                raw[i] = 1
                hold_left = hold_bars - 1

        df = pd.DataFrame(index=data.index)
        signal = pd.Series(raw, index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
