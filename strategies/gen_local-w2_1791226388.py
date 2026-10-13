from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityReliefParams:
    fast_atr: int = 5
    slow_atr: int = 30
    peak_ratio: float = 1.4
    relief_min: float = 0.08
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolatilityReliefParams]):
    strategy_id = "gen_local-w2_1791226388"

    @classmethod
    def params_type(cls) -> type[VolatilityReliefParams]:
        return VolatilityReliefParams

    @classmethod
    def warmup_bars(cls, params: VolatilityReliefParams) -> int:
        return max(params.fast_atr, params.slow_atr) + 2

    def indicators(self, data: pd.DataFrame, params: VolatilityReliefParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        fast = true_range.rolling(params.fast_atr, min_periods=params.fast_atr).mean()
        slow = true_range.rolling(params.slow_atr, min_periods=params.slow_atr).mean()
        ratio = (fast / slow.replace(0, np.nan)).fillna(0.0)
        peak = ratio.shift(1).fillna(0.0)
        earlier = ratio.shift(2).fillna(0.0)
        relief = ((peak - ratio) / peak.replace(0, np.nan)).fillna(0.0).clip(lower=0.0)
        entry = (peak >= params.peak_ratio) & (peak > earlier) & (relief >= params.relief_min) & (data["close"] > data["open"])
        entry_size = (0.25 + 2.5 * relief).clip(lower=0.25, upper=0.95)
        return pd.DataFrame({"atr_ratio": ratio, "relief": relief, "entry": entry, "entry_size": entry_size}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityReliefParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        remaining = 0
        active = False
        active_size = 0.25
        for i in range(len(data)):
            if remaining > 0:
                signal[i] = 1
                size[i] = active_size
                remaining -= 1
            elif active:
                active = False
            elif bool(indicators["entry"].iat[i]):
                active = True
                active_size = float(indicators["entry_size"].iat[i])
                signal[i] = 1
                size[i] = active_size
                remaining = params.hold_bars - 1
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
