from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 30
    deep_pct: float = 0.05
    peak_band: float = 0.005
    atr_window: int = 10
    breakeven_pct: float = 0.006
    trail_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791298548"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.lookback, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.lookback).max()
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["depth"] = (1 - data["close"] / peak.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["inside"] = ((data["high"] <= data["high"].shift(1)) &
                         (data["low"] >= data["low"].shift(1))).fillna(False)
        out["atr"] = true_range.rolling(params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0
        inside = indicators["inside"].to_numpy(dtype=bool)
        depth = indicators["depth"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry = stop = 0.0
        armed = False

        for i in range(self.warmup_bars(params), len(data)):
            if position == 0:
                if inside[i]:
                    if depth[i] >= params.deep_pct:
                        position = 1
                    elif depth[i] <= params.peak_band:
                        position = -1
                age = 0
                armed = False
            else:
                if age == 0:
                    entry = opens[i]
                    stop = entry - position * params.trail_atr * atr[i - 1]
                age += 1
                if position * (closes[i] - stop) <= 0 or age >= params.hold_bars:
                    position = 0
                else:
                    armed = armed or position * (closes[i] / entry - 1) >= params.breakeven_pct
                    if armed:
                        trailing = closes[i] - position * params.trail_atr * atr[i]
                        stop = (max(stop, entry, trailing) if position > 0
                                else min(stop, entry, trailing))
            raw[i] = position

        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
