from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    area_window: int = 126
    area_quantile: float = 0.90
    momentum_bars: int = 5
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local-w2_1791301684"

    @classmethod
    def params_type(cls):
        return PathAreaParams

    @classmethod
    def warmup_bars(cls, params):
        return max(params.area_window + 4, params.atr_window + 1, params.momentum_bars + 1)

    def indicators(self, data, params):
        price_step = np.log(data["close"]).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_step = np.log1p(data["volume"]).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        area = 0.5 * (
            price_step.shift(2) * volume_step.shift(1)
            - volume_step.shift(2) * price_step.shift(1)
            + price_step.shift(2) * volume_step
            - volume_step.shift(2) * price_step
            + price_step.shift(1) * volume_step
            - volume_step.shift(1) * price_step
        )
        threshold = area.rolling(params.area_window, min_periods=params.area_window).quantile(params.area_quantile).shift(1)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        return pd.DataFrame({
            "area": area.fillna(0.0),
            "threshold": threshold.fillna(np.inf),
            "momentum": data["close"].pct_change(params.momentum_bars).replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "atr": true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0),
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        area = indicators["area"]
        threshold = indicators["threshold"]
        entry = (area > 0) & (area > threshold) & (area.shift(1) <= threshold.shift(1)) & (indicators["momentum"] > 0)
        desired = pd.Series(0, index=data.index, dtype=int)
        active = False
        age = 0
        high_water = 0.0
        stop = -np.inf
        for i, (close, atr, enter) in enumerate(zip(data["close"], indicators["atr"], entry)):
            if not active:
                if enter:
                    active = True
                    age = 0
                    stop = -np.inf
                    desired.iat[i] = 1
                continue
            age += 1
            high_water = max(high_water, float(close)) if age > 1 else float(close)
            stop = max(stop, high_water - params.trail_atr * float(atr))
            if close < stop or age >= params.max_hold:
                active = False
                desired.iat[i] = 0
            else:
                desired.iat[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
