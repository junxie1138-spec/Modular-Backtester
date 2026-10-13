from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_pct: float = 0.005
    target_pct: float = 0.025


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714644"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return 6

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        bar_range = (data["high"] - data["low"]).clip(lower=0.0)
        avg_range = bar_range.shift(1).rolling(5, min_periods=5).mean()
        gap = data["open"] / prev_close - 1.0
        range_expansion = bar_range / avg_range.replace(0.0, np.nan)
        close_location = (data["close"] - data["low"]) / bar_range.replace(0.0, np.nan)
        return pd.DataFrame(
            {
                "gap": gap,
                "range_expansion": range_expansion,
                "close_location": close_location,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entry = (
            (indicators["gap"] >= params.gap_pct)
            & (indicators["range_expansion"] >= 1.5)
            & (indicators["close_location"] >= 0.75)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        age = 0

        for i in range(len(data)):
            if active:
                age += 1
                if high[i] >= entry_price * (1.0 + params.target_pct) or age >= 4:
                    active = False
                else:
                    raw[i] = 1
            elif entry[i] and np.isfinite(close[i]) and close[i] > 0:
                active = True
                entry_price = close[i]
                age = 0
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
