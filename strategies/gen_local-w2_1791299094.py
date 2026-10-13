from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TraversalParams:
    range_days: int = 2
    rank_window: int = 40
    rank_threshold: float = 0.8


class GeneratedStrategy(BaseStrategy[TraversalParams]):
    strategy_id = "gen_local-w2_1791299094"

    @classmethod
    def params_type(cls):
        return TraversalParams

    def warmup_bars(self, params: TraversalParams) -> int:
        return params.range_days + params.rank_window

    def indicators(self, data: pd.DataFrame, params: TraversalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["upper"] = data["high"].rolling(params.range_days).max().shift(1).fillna(data["close"])
        out["lower"] = data["low"].rolling(params.range_days).min().shift(1).fillna(data["close"])
        span = (out["upper"] - out["lower"]).replace(0, np.nan)
        traversal = ((data["close"] - data["open"]).abs() / span).fillna(0.0)
        out["rank"] = traversal.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TraversalParams,
    ) -> SignalFrame:
        midpoint = (indicators["upper"] + indicators["lower"]) / 2.0
        qualified = indicators["rank"] >= params.rank_threshold
        long = qualified & data["open"].between(indicators["lower"], midpoint) & (data["close"] > indicators["upper"])
        short = qualified & data["open"].between(midpoint, indicators["upper"]) & (data["close"] < indicators["lower"])
        df = pd.DataFrame(index=data.index)
        df["signal"] = np.select([long, short], [1, -1], default=0)
        strength = ((indicators["rank"] - params.rank_threshold) / (1.0 - params.rank_threshold)).clip(0.0, 1.0)
        df["size"] = (0.25 + 0.75 * strength).shift(1).fillna(0.25).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
