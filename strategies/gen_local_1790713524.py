from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    scale_window: int = 20
    percentile_window: int = 63
    lower_percentile: float = 0.15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713524"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return params.ma_window + params.scale_window + params.percentile_window

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        distance = data["close"] - data["close"].rolling(params.ma_window).mean()
        scale = distance.rolling(params.scale_window).std(ddof=0).replace(0, np.nan)
        zscore = distance / scale
        threshold = zscore.rolling(params.percentile_window).quantile(params.lower_percentile).shift(1)
        return pd.DataFrame({"zscore": zscore, "threshold": threshold}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        condition = (indicators["zscore"] <= indicators["threshold"]).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
