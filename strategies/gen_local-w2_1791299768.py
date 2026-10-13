from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignaturePositionParams:
    fast_window: int = 5
    slow_window: int = 20
    loop_bars: int = 5
    percentile_window: int = 40
    area_percentile: float = 0.70


class GeneratedStrategy(BaseStrategy[SignaturePositionParams]):
    strategy_id = "gen_local-w2_1791299768"

    @classmethod
    def params_type(cls) -> type[SignaturePositionParams]:
        return SignaturePositionParams

    def warmup_bars(self, params: SignaturePositionParams) -> int:
        return max(params.fast_window, params.slow_window) + params.loop_bars + params.percentile_window + 1

    def indicators(self, data: pd.DataFrame, params: SignaturePositionParams) -> pd.DataFrame:
        def position(window: int) -> pd.Series:
            floor = data["low"].rolling(window, min_periods=window).min()
            ceiling = data["high"].rolling(window, min_periods=window).max()
            span = (ceiling - floor).replace(0, np.nan)
            return ((data["close"] - floor) / span).clip(0, 1).fillna(0.5)

        x = position(params.slow_window)
        y = position(params.fast_window)
        segment = x.shift(1) * y - y.shift(1) * x
        area = (segment.rolling(params.loop_bars, min_periods=params.loop_bars).sum()
                + x * y.shift(params.loop_bars)
                - y * x.shift(params.loop_bars)) / 2
        area = area.fillna(0.0)
        past = area.shift(1)
        upper = past.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.area_percentile)
        lower = past.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(1 - params.area_percentile)
        return pd.DataFrame({"area": area, "upper": upper, "lower": lower}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignaturePositionParams,
    ) -> SignalFrame:
        eligible = np.arange(len(data)) >= self.warmup_bars(params)
        long_entry = eligible & (indicators["area"] > 0) & (indicators["area"] > indicators["upper"])
        short_entry = eligible & (indicators["area"] < 0) & (indicators["area"] < indicators["lower"])
        signal = []
        state = 0
        for buy, sell in zip(long_entry, short_entry):
            if state == 1 and sell:
                state = 0
            elif state == -1 and buy:
                state = 0
            elif state == 0:
                state = 1 if buy else -1 if sell else 0
            signal.append(state)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
