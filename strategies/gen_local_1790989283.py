from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeElasticityParams:
    range_window: int = 20
    autocorr_window: int = 40
    autocorr_ceiling: float = -0.15
    compression_ratio: float = 0.85
    minimum_close_location: float = 0.60
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeElasticityParams]):
    strategy_id = "gen_local_1790989283"

    @classmethod
    def params_type(cls) -> type[RangeElasticityParams]:
        return RangeElasticityParams

    @staticmethod
    def warmup_bars(params: RangeElasticityParams) -> int:
        return int(max(params.range_window, params.autocorr_window + 3, 3))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeElasticityParams) -> pd.DataFrame:
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        relative_range = width / data["close"].where(data["close"] > 0)
        log_range = np.log(relative_range)
        range_change = log_range.diff()
        mean_range = relative_range.rolling(
            params.range_window, min_periods=params.range_window
        ).mean()
        range_autocorr = range_change.rolling(
            params.autocorr_window, min_periods=params.autocorr_window
        ).corr(range_change.shift(1)).shift(1)
        result = pd.DataFrame({
            "relative_range": relative_range,
            "log_range": log_range,
            "range_change": range_change,
            "mean_range": mean_range,
            "range_ratio": relative_range / mean_range,
            "range_autocorr": range_autocorr,
            "close_location": (data["close"] - data["low"]) / width,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeElasticityParams,
    ) -> SignalFrame:
        qualifies = (
            indicators["range_autocorr"].lt(params.autocorr_ceiling)
            & indicators["range_ratio"].lt(params.compression_ratio)
            & indicators["close_location"].ge(params.minimum_close_location)
        ).fillna(False)
        confirmed = qualifies & qualifies.shift(1, fill_value=False)
        entries = confirmed.to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        exit_bar = -1
        holding_bars = max(1, int(params.holding_bars))
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    targets[i] = 1
                else:
                    exit_bar = -1
                continue
            if i >= warmup - 1 and entries[i]:
                targets[i] = 1
                exit_bar = i + holding_bars
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
