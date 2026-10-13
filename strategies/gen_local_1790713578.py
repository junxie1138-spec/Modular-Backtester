from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    drawdown_pct: float = 0.025
    target_pct: float = 0.02
    max_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713578"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return 8

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        fast = close.ewm(span=3, adjust=False, min_periods=3).mean()
        slow = close.ewm(span=8, adjust=False, min_periods=8).mean()
        slope = fast - fast.shift(1)
        innovation = slope - slope.shift(1)
        peak = close.rolling(7, min_periods=7).max()
        drawdown = close / peak - 1.0
        return pd.DataFrame({"slope": slope, "innovation": innovation, "drawdown": drawdown}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        valid = indicators[["slope", "innovation", "drawdown"]].notna().all(axis=1)
        entries = (valid & (indicators["drawdown"] <= -params.drawdown_pct) & (indicators["slope"] > 0) & (indicators["innovation"] > 0)).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if active:
                held = i - entry_bar
                if close[i] >= entry_price * (1.0 + params.target_pct) or held >= params.max_bars:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                entry_price = close[i]
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
