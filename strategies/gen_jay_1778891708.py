from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778891708"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.lookback

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        hl = data["high"] - data["low"]
        ind["hl_range"] = hl
        ind["range_p80"] = hl.rolling(params.lookback, min_periods=params.lookback).quantile(0.80)
        ind["bar_dir"] = np.sign(data["close"] - data["open"])
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        hl = indicators["hl_range"]
        p80 = indicators["range_p80"]
        bar_dir = indicators["bar_dir"]

        # Range spike: current range exceeds 80th percentile of recent history
        spike = (hl > p80) & p80.notna()

        # Fade the bar's close direction on range spikes (mean reversion)
        raw = pd.Series(0, index=data.index, dtype=int)
        raw[spike & (bar_dir > 0)] = -1   # large up bar -> short
        raw[spike & (bar_dir < 0)] = 1    # large down bar -> long

        # Fixed-bar exit: hold exactly hold_bars bars, then exit
        raw_vals = raw.to_numpy()
        n = len(raw_vals)
        hold = params.hold_bars
        out = np.zeros(n, dtype=np.int64)

        i = 0
        while i < n:
            entry = int(raw_vals[i])
            if entry != 0:
                end = min(i + hold, n)
                out[i:end] = entry
                i = i + hold + 1  # advance past hold period and implicit exit bar
            else:
                i += 1

        df["signal"] = pd.Series(out, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
