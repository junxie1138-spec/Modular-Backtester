from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownShelfParams:
    peak_window: int = 30
    shelf_window: int = 6
    history_window: int = 126
    compression_pct: float = 0.20
    depth_pct: float = 0.70
    profit_target: float = 0.035
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownShelfParams]):
    strategy_id = "gen_local_1791296246"

    @classmethod
    def params_type(cls):
        return DrawdownShelfParams

    def warmup_bars(self, params: DrawdownShelfParams) -> int:
        return params.peak_window + params.shelf_window + params.history_window + 2

    def indicators(self, data: pd.DataFrame, params: DrawdownShelfParams) -> pd.DataFrame:
        depth = 1.0 - data["close"] / data["close"].rolling(params.peak_window).max()
        shelf = depth.rolling(params.shelf_window).max() - depth.rolling(params.shelf_window).min()
        out = pd.DataFrame(index=data.index)
        out["depth"] = depth
        out["prior_depth"] = depth.shift(1)
        out["prior_shelf"] = shelf.shift(1)
        out["shelf_cut"] = shelf.rolling(params.history_window).quantile(params.compression_pct).shift(1)
        out["deep_cut"] = depth.rolling(params.history_window).quantile(params.depth_pct).shift(1)
        out["floor"] = depth.shift(1).rolling(params.shelf_window).min()
        out["ceiling"] = depth.shift(1).rolling(params.shelf_window).max()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownShelfParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        ready = valid & (indicators["prior_shelf"] <= indicators["shelf_cut"]) & (indicators["prior_depth"] >= indicators["deep_cut"])
        long_entry = ready & (indicators["depth"] < indicators["floor"])
        short_entry = ready & (indicators["depth"] > indicators["ceiling"])
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0
        position = 0
        entry_bar = -1
        entry_open = 0.0
        for i in range(len(data)):
            if position and entry_bar < 0:
                entry_bar = i
                entry_open = float(data["open"].iat[i])
            if position:
                gain = position * (float(data["close"].iat[i]) / entry_open - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_bar = -1
            elif long_entry.iat[i]:
                position = 1
            elif short_entry.iat[i]:
                position = -1
            df["signal"].iat[i] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
