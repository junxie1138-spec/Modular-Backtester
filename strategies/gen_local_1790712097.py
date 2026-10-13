from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_lookback: int = 5
    oversold_threshold: float = 0.02
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712097"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(200, params.roc_lookback + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        roc = data["close"].pct_change(params.roc_lookback)
        return pd.DataFrame(
            {
                "roc": roc,
                "roc_acceleration": roc.diff(),
                "ma_200": data["close"].rolling(200, min_periods=200).mean(),
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
            (indicators["roc"] <= -params.oversold_threshold)
            & (indicators["roc_acceleration"] > 0)
            & (data["close"] > indicators["ma_200"])
        ).fillna(False).to_numpy(dtype=bool)

        raw = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, is_entry in enumerate(entry):
            if remaining > 0:
                raw[i] = 1
                remaining -= 1
            elif is_entry:
                raw[i] = 1
                remaining = max(0, params.hold_bars - 1)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
