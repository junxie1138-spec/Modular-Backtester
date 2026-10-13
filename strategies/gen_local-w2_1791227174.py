from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownAutocorrParams:
    peak_window: int = 20
    autocorr_window: int = 12
    min_depth: float = 0.015
    min_change: float = 0.002
    max_autocorr: float = -0.15


class GeneratedStrategy(BaseStrategy[DrawdownAutocorrParams]):
    strategy_id = "gen_local-w2_1791227174"

    @classmethod
    def params_type(cls) -> type[DrawdownAutocorrParams]:
        return DrawdownAutocorrParams

    @classmethod
    def warmup_bars(cls, params: DrawdownAutocorrParams) -> int:
        return params.peak_window + params.autocorr_window + 1

    def indicators(self, data: pd.DataFrame, params: DrawdownAutocorrParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = 1.0 - data["close"] / peak
        change = depth.diff()
        autocorr = change.rolling(params.autocorr_window, min_periods=params.autocorr_window).corr(change.shift(1))
        return pd.DataFrame({
            "depth": depth.fillna(0.0),
            "depth_change": change.fillna(0.0),
            "depth_autocorr": autocorr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownAutocorrParams,
    ) -> SignalFrame:
        eligible = (indicators["depth"] >= params.min_depth) & (indicators["depth_autocorr"] <= params.max_autocorr)
        long_entry = eligible & (indicators["depth_change"] >= params.min_change)
        short_entry = eligible & (indicators["depth_change"] <= -params.min_change)

        position = 0
        states = []
        for go_long, go_short in zip(long_entry, short_entry):
            if position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            elif position == 0:
                position = int(go_long) - int(go_short)
            states.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = states
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
