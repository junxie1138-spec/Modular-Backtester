from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    analog_window: int = 126
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791224808"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.analog_window + 68

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        shape = pd.concat([returns.shift(j) for j in range(5)], axis=1)
        shape = shape.div(shape.abs().sum(axis=1).replace(0, np.nan), axis=0)
        distances = pd.concat(
            [(shape - shape.shift(lag)).abs().sum(axis=1, min_count=5)
             for lag in range(21, params.analog_window + 1)],
            axis=1,
        )
        nearest = distances.min(axis=1)
        nearest = nearest.where(shape.shift(params.analog_window).notna().all(axis=1))

        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)

        out = pd.DataFrame(index=data.index)
        out["stretch_rank"] = close.pct_change(20, fill_method=None).rolling(params.analog_window).rank(pct=True)
        out["familiarity_rank"] = nearest.rolling(63).rank(pct=True)
        out["atr"] = true_range.rolling(14).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        rank = indicators["stretch_rank"].to_numpy()
        familiarity = indicators["familiarity_rank"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        high_water = low_water = stop = 0.0
        held = 0

        for i, price in enumerate(close):
            if position == 0:
                if (np.isfinite(rank[i]) and np.isfinite(familiarity[i])
                        and np.isfinite(atr[i]) and familiarity[i] <= 0.25):
                    position = 1 if rank[i] <= 0.10 else -1 if rank[i] >= 0.90 else 0
                    if position:
                        high_water = low_water = price
                        stop = (price - params.trail_atr * atr[i] if position == 1
                                else price + params.trail_atr * atr[i])
                        held = 0
            else:
                held += 1
                if position == 1:
                    high_water = max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if price <= stop or held >= 20:
                        position = 0
                else:
                    low_water = min(low_water, price)
                    stop = min(stop, low_water + params.trail_atr * atr[i])
                    if price >= stop or held >= 20:
                        position = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
