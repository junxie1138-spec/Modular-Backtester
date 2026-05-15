from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778883971"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.lookback + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        hl_range = data["high"] - data["low"]
        roll_median = hl_range.rolling(params.lookback).median()

        # Compressed bar: range is more than 20% below rolling median (low noise)
        is_compressed = (hl_range < roll_median * 0.80).fillna(False).astype(float)

        # SNR proxy: close position within the bar range (1 = at high, 0 = at low)
        range_span = hl_range.replace(0, np.nan)
        range_pos = (data["close"] - data["low"]) / range_span

        # Sustained compression: both today and yesterday were compressed
        consec_compressed = is_compressed.rolling(2).sum()

        ind["consec_compressed"] = consec_compressed
        ind["range_pos"] = range_pos
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        sig = np.zeros(n, dtype=int)

        # Entry: two consecutive compressed bars with close in upper 45% of range
        entry_raw = (
            (indicators["consec_compressed"].fillna(0) >= 2.0)
            & (indicators["range_pos"].fillna(0.0) > 0.55)
        ).to_numpy()

        hold = params.hold_bars
        i = 0
        while i < n:
            if entry_raw[i]:
                end = min(i + hold, n)
                sig[i:end] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(
            {"signal": sig, "size": np.ones(n, dtype=float)},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
