from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDecayParams:
    gap_window: int = 63
    gap_quantile: float = 0.80
    profit_target: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapDecayParams]):
    strategy_id = "gen_local-w2_1791296868"

    @classmethod
    def params_type(cls):
        return GapDecayParams

    def warmup_bars(self, params):
        return int(params.gap_window + 3)

    def indicators(self, data, params):
        gap = (data["open"] / data["close"].shift(1) - 1.0).fillna(0.0)
        cutoff = gap.abs().shift(1).rolling(params.gap_window, min_periods=params.gap_window).quantile(params.gap_quantile)
        first = gap.shift(2)
        second = gap.shift(1)
        rejected = ((gap > 0) & (data["close"] < data["open"])) | ((gap < 0) & (data["close"] > data["open"]))
        entry = (
            cutoff.shift(2).notna()
            & (first.abs() > cutoff.shift(2))
            & (first * second > 0)
            & (second * gap > 0)
            & (first.abs() > second.abs())
            & (second.abs() > gap.abs())
            & rejected
        )
        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_cutoff"] = cutoff.fillna(0.0)
        out["entry_side"] = np.where(entry, -np.sign(gap), 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params: GapDecayParams) -> SignalFrame:
        count = len(data)
        targets = np.zeros(count, dtype=int)
        entries = indicators["entry_side"].to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = 0.0
        entry_bar = 0

        for i in range(count):
            if i and targets[i - 1] != position:
                position = targets[i - 1]
                if position:
                    entry_price = opens[i]
                    entry_bar = i
            if position:
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    targets[i] = 0
                else:
                    targets[i] = position
            else:
                targets[i] = entries[i]

        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
