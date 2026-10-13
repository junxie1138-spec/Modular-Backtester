from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    shape_bars: int = 12
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791296080"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 12 + params.shape_bars + 252 + 20

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weight = 1.0
        fd = pd.Series(0.0, index=data.index)
        for lag in range(12):
            if lag:
                weight *= (lag - 1 - 0.4) / lag
            fd = fd.add(weight * close.shift(lag), fill_value=0.0)
        out["fd_close"] = fd.fillna(0.0)

        previous = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean().fillna(0.0)

        length = params.shape_bars
        values = out["fd_close"].to_numpy(dtype=float)
        patterns = np.lib.stride_tricks.sliding_window_view(values, length)
        means = patterns.mean(axis=1, keepdims=True)
        scales = patterns.std(axis=1, keepdims=True)
        patterns = (patterns - means) / np.maximum(scales, 1e-12)
        prices = close.to_numpy(dtype=float)
        forecast = np.zeros(len(data), dtype=int)
        for i in range(self.warmup_bars(params), len(data)):
            ends = np.arange(max(length + 10, i - 252), i - 19)
            distances = np.mean((patterns[ends - length + 1] - patterns[i - length + 1]) ** 2, axis=1)
            nearest = ends[np.argpartition(distances, 6)[:7]]
            analog_return = np.median(prices[nearest + 20] / prices[nearest] - 1.0)
            forecast[i] = int(analog_return > 0.01) - int(analog_return < -0.01)
        out["forecast"] = forecast
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=int)
        position = 0
        target = 0
        entry = stop = 0.0
        age = 0
        armed = False

        for i in range(len(data)):
            if position != target:
                position = target
                if position:
                    entry = opening[i]
                    stop = entry - position * params.trail_atr * atr[i]
                    age = 0
                    armed = False

            if position:
                age += 1
                if position * (closing[i] - stop) <= 0 or age >= 20:
                    target = 0
                else:
                    if position * (closing[i] / entry - 1.0) >= 0.025:
                        armed = True
                    if armed:
                        candidate = closing[i] - position * params.trail_atr * atr[i]
                        if position > 0:
                            stop = max(stop, entry, candidate)
                        else:
                            stop = min(stop, entry, candidate)
                    target = position
            elif forecast[i]:
                target = int(forecast[i])
            raw[i] = target

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
