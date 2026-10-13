from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownPositionParams:
    lookback: int = 30
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[DrawdownPositionParams]):
    strategy_id = "gen_local-w2_1791301152"

    @classmethod
    def params_type(cls):
        return DrawdownPositionParams

    def warmup_bars(self, params: DrawdownPositionParams) -> int:
        return max(2 * params.lookback, 15)

    def indicators(self, data: pd.DataFrame, params: DrawdownPositionParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        depth = (1.0 - close / close.rolling(params.lookback, min_periods=params.lookback).max()).clip(lower=0.0)
        peak_depth = depth.rolling(params.lookback, min_periods=params.lookback).max()
        out["position"] = (depth / peak_depth.replace(0.0, np.nan)).fillna(0.0)
        out["peak_depth"] = peak_depth.fillna(0.0)
        out["atr"] = true_range.rolling(14, min_periods=14).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownPositionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        position = indicators["position"].to_numpy()
        peak_depth = indicators["peak_depth"].to_numpy()
        atr = indicators["atr"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        side = 0
        bars = 0
        high_water = low_water = stop = 0.0

        for i in range(1, len(data)):
            if i < self.warmup_bars(params):
                continue
            price = close[i]
            if side == 1:
                bars += 1
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if price <= stop or bars >= 10:
                    side = 0
            elif side == -1:
                bars += 1
                low_water = min(low_water, price)
                stop = min(stop, low_water + params.trail_atr * atr[i])
                if price >= stop or bars >= 10:
                    side = 0
            elif atr[i] > 0 and price > 0 and peak_depth[i] >= 2.0 * atr[i] / price:
                if position[i - 1] > 0.55 and position[i] <= 0.55:
                    side = 1
                    bars = 0
                    high_water = price
                    stop = price - params.trail_atr * atr[i]
                elif position[i - 1] < 0.45 and position[i] >= 0.45:
                    side = -1
                    bars = 0
                    low_water = price
                    stop = price + params.trail_atr * atr[i]
            signals[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
