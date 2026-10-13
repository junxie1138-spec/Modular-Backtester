from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalSaturationParams:
    novelty_ceiling: float = 0.25
    hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[OrdinalSaturationParams]):
    strategy_id = "gen_local_1791297968"

    @classmethod
    def params_type(cls) -> type[OrdinalSaturationParams]:
        return OrdinalSaturationParams

    def warmup_bars(self, params: OrdinalSaturationParams) -> int:
        return 43

    def indicators(self, data: pd.DataFrame, params: OrdinalSaturationParams) -> pd.DataFrame:
        close = data["close"]
        pattern = pd.Series(0, index=data.index, dtype=int)
        bit = 0
        for older in range(1, 4):
            for newer in range(older):
                pattern += (close.shift(older) >= close.shift(newer)).astype(int) * (1 << bit)
                bit += 1
        pattern = pattern.where(close.shift(3).notna())
        seen = pd.concat(
            [pattern.eq(pattern.shift(lag)) for lag in range(1, 21)], axis=1
        ).any(axis=1)
        novelty = (~seen).astype(float).where(pattern.shift(20).notna())
        result = pd.DataFrame(index=data.index)
        result["ordinal_pattern"] = pattern
        result["novelty"] = novelty
        result["novelty_rate"] = novelty.rolling(20, min_periods=20).mean()
        result["trend_20"] = close.div(close.shift(20)).sub(1)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalSaturationParams,
    ) -> SignalFrame:
        eligible = (
            indicators["novelty_rate"].le(params.novelty_ceiling)
            & indicators["trend_20"].gt(0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0
        entry_bar = -1
        for i in range(len(df)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    df.iat[i, 0] = 1
            elif eligible.iat[i]:
                entry_bar = i
                df.iat[i, 0] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
