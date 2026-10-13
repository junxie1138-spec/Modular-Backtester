from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    vol_percentile: float = 0.85
    profit_target: float = 0.03
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791226161"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 146

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_long = true_range.rolling(20, min_periods=20).mean()
        ratio = (
            true_range.rolling(4, min_periods=4).mean()
            / atr_long.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        recent_low = data["low"].rolling(10, min_periods=10).min()
        recent_high = data["high"].rolling(10, min_periods=10).max()
        out = pd.DataFrame(index=data.index)
        out["vol_ratio"] = ratio.fillna(0.0)
        out["vol_threshold"] = (
            ratio.rolling(126, min_periods=126)
            .quantile(params.vol_percentile)
            .shift(1)
            .fillna(np.inf)
        )
        out["range_position"] = (
            (data["close"] - recent_low)
            / (recent_high - recent_low).replace(0, np.nan)
        ).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry = (
            (indicators["vol_ratio"] > indicators["vol_threshold"])
            & indicators["range_position"].between(0.55, 0.85)
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        entry_fill = -1
        exit_fill = -1
        entry_price = 0.0
        entry_bar = -1
        holding = False

        for i in range(len(data)):
            if i == exit_fill:
                holding = False
                exit_fill = -1
            if i == entry_fill:
                holding = True
                entry_price = float(opens[i])
                entry_bar = i
                entry_fill = -1

            if holding:
                if exit_fill < 0 and (
                    closes[i] >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar >= params.hold_bars
                ):
                    exit_fill = i + 2
                decisions[i] = int(exit_fill < 0)
            elif entry_fill >= 0:
                decisions[i] = 1
            elif entry[i] and i + 2 < len(data):
                entry_fill = i + 2
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
