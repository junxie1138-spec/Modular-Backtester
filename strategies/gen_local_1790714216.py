from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    corr_bars: int = 20
    direction_bars: int = 8
    hold_bars: int = 16
    corr_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714216"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(200, params.corr_bars + 2, params.direction_bars)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        span = (data["high"] - data["low"]).clip(lower=0.0)
        normalized_range = span / close.replace(0.0, np.nan)
        range_change = normalized_range.pct_change()
        prior = range_change.shift(1)
        corr = range_change.rolling(params.corr_bars, min_periods=params.corr_bars).corr(prior)
        candle_direction = ((2.0 * close - data["high"] - data["low"]) / span.replace(0.0, np.nan)).fillna(0.0)
        directional_imbalance = candle_direction.rolling(params.direction_bars, min_periods=params.direction_bars).mean()
        trend = close.rolling(200, min_periods=200).mean()
        return pd.DataFrame(
            {"range_autocorr": corr, "directional_imbalance": directional_imbalance, "trend": trend},
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
        corr = indicators["range_autocorr"].to_numpy()
        imbalance = indicators["directional_imbalance"].to_numpy()
        close = data["close"].to_numpy()
        trend = indicators["trend"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0

        for i in range(len(data)):
            if position:
                age += 1
                if age >= params.hold_bars:
                    position = 0
                    age = 0
            elif (
                np.isfinite(corr[i])
                and np.isfinite(imbalance[i])
                and np.isfinite(trend[i])
                and corr[i] >= params.corr_threshold
            ):
                if close[i] > trend[i] and imbalance[i] > 0:
                    position = 1
                    age = 0
                elif close[i] < trend[i] and imbalance[i] < 0:
                    position = -1
                    age = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
