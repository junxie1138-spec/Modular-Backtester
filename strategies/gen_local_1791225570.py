from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRangeParams:
    range_lookback: int = 5
    reversal_threshold: float = 0.2


class GeneratedStrategy(BaseStrategy[GapRangeParams]):
    strategy_id = "gen_local_1791225570"

    @classmethod
    def params_type(cls) -> type[GapRangeParams]:
        return GapRangeParams

    @classmethod
    def warmup_bars(cls, params: GapRangeParams) -> int:
        return params.range_lookback + 1

    def indicators(self, data: pd.DataFrame, params: GapRangeParams) -> pd.DataFrame:
        daily_range = data["high"] - data["low"]
        prior_range = daily_range.shift(1)
        baseline = prior_range.rolling(params.range_lookback).mean().replace(0, np.nan)
        gap = data["open"] - data["close"].shift(1)
        out = pd.DataFrame(index=data.index)
        out["gap"] = gap.fillna(0.0)
        out["balance"] = ((gap + prior_range - daily_range) / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRangeParams,
    ) -> SignalFrame:
        enter = (indicators["gap"] > 0) & (indicators["balance"] >= params.reversal_threshold)
        exit_ = indicators["balance"] <= -params.reversal_threshold
        desired = pd.Series(np.nan, index=data.index)
        desired.loc[enter] = 1
        desired.loc[exit_] = 0
        df = pd.DataFrame(index=data.index)
        df["signal"] = desired.ffill().fillna(0).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
