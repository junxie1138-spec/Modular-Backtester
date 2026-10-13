from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_lookback: int = 60
    gap_percentile: float = 0.90
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712144"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(201, params.gap_lookback + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        overnight_gap = data["open"] / data["close"].shift(1) - 1.0
        gap_rank = overnight_gap.abs().rolling(params.gap_lookback, min_periods=params.gap_lookback).rank(pct=True)
        trend = data["close"].rolling(200, min_periods=200).mean()
        return pd.DataFrame({"overnight_gap": overnight_gap, "gap_rank": gap_rank, "trend_200": trend}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        eligible = (
            indicators["gap_rank"].ge(params.gap_percentile)
            & indicators["overnight_gap"].lt(0)
            & data["close"].gt(data["open"])
            & data["close"].gt(indicators["trend_200"])
        ).fillna(False)
        raw = eligible.astype(int)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, entry in enumerate(raw.to_numpy()):
            if remaining > 0:
                remaining -= 1
            elif entry:
                signal[i] = 1
                remaining = max(0, params.hold_bars - 1)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
