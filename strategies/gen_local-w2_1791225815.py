from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    path_bars: int = 4
    rank_bars: int = 60
    rank_quantile: float = 0.8
    breakout_bars: int = 18
    atr_bars: int = 14
    trail_atr: float = 1.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791225815"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.path_bars + params.rank_bars, params.breakout_bars + 1, params.atr_bars)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff().abs()
        path = change.rolling(params.path_bars).sum()
        efficiency = (data["close"].diff(params.path_bars).abs() / path.replace(0, np.nan)).fillna(0.0)
        out["efficiency"] = efficiency
        out["rank_cutoff"] = efficiency.rolling(params.rank_bars).quantile(params.rank_quantile).fillna(1.1)
        out["prior_high"] = data["high"].rolling(params.breakout_bars).max().shift(1).fillna(np.inf)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_bars).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        entry = (
            (data["close"] > indicators["prior_high"])
            & (indicators["efficiency"] >= indicators["rank_cutoff"])
        ).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        held = False
        high_water = stop = 0.0
        age = 0

        for i in range(self.warmup_bars(params), len(data)):
            if held:
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.max_hold:
                    held = False
            elif entry[i]:
                held = True
                high_water = close[i]
                stop = high_water - params.trail_atr * atr[i]
                age = 0
            signals[i] = int(held)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
