from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityStretchParams:
    lookback: int = 20
    stretch: float = 1.5


class GeneratedStrategy(BaseStrategy[VolatilityStretchParams]):
    strategy_id = "gen_local_1790714347"

    @classmethod
    def params_type(cls):
        return VolatilityStretchParams

    @classmethod
    def warmup_bars(cls, params: VolatilityStretchParams) -> int:
        return params.lookback + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: VolatilityStretchParams) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(params.lookback, min_periods=params.lookback).mean()
        vol = close.pct_change().rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        return pd.DataFrame({"mean": mean, "vol": vol}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityStretchParams) -> SignalFrame:
        close = data["close"]
        distance = (close / indicators["mean"] - 1.0) / indicators["vol"].replace(0.0, np.nan)
        raw = (distance <= -params.stretch).fillna(False).astype(int)
        signal = raw.shift(1).fillna(0).astype(int)
        out = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        position = 0
        entry = 0.0
        age = 0
        for i in range(len(out)):
            price = float(close.iloc[i])
            if position:
                age += 1
                if price >= entry * 1.02 or age >= 4:
                    position = 0
            if not position and signal.iloc[i] == 1:
                position = 1
                entry = price
                age = 0
            out.iloc[i, 0] = position
        out["signal"] = out["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
