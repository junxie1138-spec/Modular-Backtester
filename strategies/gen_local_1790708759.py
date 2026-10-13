from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRangeRejectionParams:
    gap_atr: float = 0.5
    atr_len: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapRangeRejectionParams]):
    strategy_id = "gen_local_1790708759"

    @classmethod
    def params_type(cls):
        return GapRangeRejectionParams

    def warmup_bars(self, params: GapRangeRejectionParams) -> int:
        return max(200, params.atr_len + 1)

    def indicators(self, data: pd.DataFrame, params: GapRangeRejectionParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        prev_low = data["low"].shift(1)
        tr = pd.concat([data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()
        prior_atr = atr.shift(1)
        ma = data["close"].rolling(200, min_periods=200).mean()
        entry = (ma.notna() & prior_atr.notna() & (data["close"] > ma) & (prev_close - data["open"] >= params.gap_atr * prior_atr) & (data["open"] < prev_low) & (data["close"] > prev_low) & (data["close"] < prev_close))
        return pd.DataFrame({"entry": entry.fillna(False), "atr": atr.fillna(0.0), "ma200": ma.fillna(0.0)}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRangeRejectionParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        entry_setup = indicators["entry"].to_numpy(dtype=bool)
        position = np.zeros(len(data), dtype=int)
        active = False
        entry = peak = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if active:
                if low[i] <= stop:
                    active = False
                else:
                    peak = max(peak, high[i])
                    if peak >= entry * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if close[i] <= stop or close[i] < ma[i] or i - entry_bar >= params.max_hold:
                        active = False
            elif entry_setup[i]:
                active = True
                entry = peak = close[i]
                stop = entry - params.trail_atr * atr[i]
                entry_bar = i
            position[i] = int(active)
        df = pd.DataFrame({"signal": position, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
