from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    volume_window: int = 20
    baseline_span: int = 10
    switch_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791301939"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.volume_window + 1, params.baseline_span)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        daily_range = (data["high"] - data["low"]).replace(0, np.nan)
        prior_volume = data["volume"].rolling(params.volume_window).mean().shift(1).replace(0, np.nan)
        impulse = (
            (data["close"].diff() / daily_range).clip(-2, 2)
            * (data["volume"] / prior_volume).clip(upper=3)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        edge = impulse.ewm(span=2, adjust=False).mean() - impulse.ewm(span=params.baseline_span, adjust=False).mean()
        return pd.DataFrame({"impulse": impulse, "edge": edge.fillna(0.0)}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        up = (indicators["edge"] > params.switch_threshold) & (indicators["impulse"] > 0)
        down = (indicators["edge"] < -params.switch_threshold) & (indicators["impulse"] < 0)
        long_entry = up & up.shift(1, fill_value=False)
        short_entry = down & down.shift(1, fill_value=False)

        positions = []
        position = 0
        for buy, sell in zip(long_entry, short_entry):
            if position == 1 and sell:
                position = 0
            elif position == -1 and buy:
                position = 0
            elif position == 0:
                if buy:
                    position = 1
                elif sell:
                    position = -1
            positions.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
