from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBodyParams:
    peak_window: int = 10
    min_depth: float = 0.001


class GeneratedStrategy(BaseStrategy[DrawdownBodyParams]):
    strategy_id = "gen_local-w2_1791223693"

    @classmethod
    def params_type(cls):
        return DrawdownBodyParams

    def warmup_bars(self, params: DrawdownBodyParams) -> int:
        return params.peak_window + 1

    def indicators(self, data: pd.DataFrame, params: DrawdownBodyParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - data["close"] / peak.where(peak > 0)).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ready"] = depth.notna() & depth.shift(1).notna()
        out["depth"] = depth.fillna(0.0)
        out["depth_change"] = depth.diff().fillna(0.0)
        out["body"] = (data["close"] - data["open"]).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownBodyParams,
    ) -> SignalFrame:
        active = indicators["ready"] & (indicators["depth"] >= params.min_depth)
        long_entry = (active & (indicators["depth_change"] < 0) & (indicators["body"] > 0)).to_numpy()
        short_entry = (active & (indicators["depth_change"] > 0) & (indicators["body"] < 0)).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (go_long, go_short) in enumerate(zip(long_entry, short_entry)):
            if position == 0:
                if go_long:
                    position = 1
                elif go_short:
                    position = -1
            elif (position == 1 and go_short) or (position == -1 and go_long):
                position = 0
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
