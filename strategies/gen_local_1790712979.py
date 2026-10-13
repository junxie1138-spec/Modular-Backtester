from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_window: int = 10
    compression_percentile: float = 0.25
    gap_threshold: float = 0.002
    regime_window: int = 200


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712979"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.compression_window * 2, params.regime_window) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prior_close = data["close"].shift(1)
        day_range = (data["high"] - data["low"]) / prior_close.replace(0, np.nan)
        range_mean = day_range.rolling(params.compression_window, min_periods=params.compression_window).mean()
        compression_cutoff = range_mean.rolling(
            params.compression_window,
            min_periods=params.compression_window,
        ).quantile(params.compression_percentile)
        return pd.DataFrame(
            {
                "gap": data["open"] / prior_close - 1.0,
                "compressed": range_mean.le(compression_cutoff),
                "ma200": data["close"].rolling(
                    params.regime_window,
                    min_periods=params.regime_window,
                ).mean(),
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
        gap = indicators["gap"]
        up_entry = (
            indicators["compressed"]
            & gap.ge(params.gap_threshold)
            & data["close"].gt(indicators["ma200"])
        ).fillna(False)
        down_entry = (
            indicators["compressed"]
            & gap.le(-params.gap_threshold)
            & data["close"].lt(indicators["ma200"])
        ).fillna(False)

        raw = pd.Series(0, index=data.index, dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1:
                if down_entry.iat[i]:
                    position = -1
                elif up_entry.iat[i]:
                    position = 1
                else:
                    position = 0
            elif position == -1:
                if up_entry.iat[i]:
                    position = 1
                elif down_entry.iat[i]:
                    position = -1
                else:
                    position = 0
            elif up_entry.iat[i]:
                position = 1
            elif down_entry.iat[i]:
                position = -1
            raw.iat[i] = position

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
