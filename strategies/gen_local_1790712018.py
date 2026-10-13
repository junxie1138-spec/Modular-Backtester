from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoilParams:
    atr_period: int = 14
    min_gap_atr: float = 0.4
    trail_k: float = 1.5


class GeneratedStrategy(BaseStrategy[GapRecoilParams]):
    strategy_id = "gen_local_1790712018"

    @classmethod
    def params_type(cls):
        return GapRecoilParams

    @staticmethod
    def warmup_bars(params: GapRecoilParams) -> int:
        return int(params.atr_period) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRecoilParams) -> pd.DataFrame:
        close = data["close"]
        prior_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(int(params.atr_period), min_periods=int(params.atr_period)).mean()
        gap = (prior_close - data["open"]).clip(lower=0.0)
        ratio = gap.div(atr.where(atr > 0.0))
        reclaimed = (data["close"] > data["open"]) & (data["close"] > prior_close)
        entry = (reclaimed & (ratio >= float(params.min_gap_atr))).fillna(False)
        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["entry"] = entry.astype(bool)
        out["gap_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRecoilParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=bool)
        gap_ratio = indicators["gap_ratio"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.01, dtype=float)
        position = False
        high_water = 0.0
        bars_held = 0
        for i in range(len(data)):
            if not position:
                if entry[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    position = True
                    high_water = close[i]
                    bars_held = 0
                    signal[i] = 1
                    size[i] = min(1.0, max(0.01, gap_ratio[i] / max(float(params.min_gap_atr), 1e-9)))
            else:
                bars_held += 1
                high_water = max(high_water, close[i])
                stop = high_water - float(params.trail_k) * atr[i]
                if not np.isfinite(atr[i]) or close[i] <= stop or bars_held >= 2:
                    position = False
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = size[i - 1]
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
