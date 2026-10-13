from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_threshold: float = 0.85
    profit_target: float = 0.05


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791225830"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 50

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
        short_atr = true_range.rolling(10, min_periods=10).mean()
        long_atr = true_range.rolling(50, min_periods=50).mean()
        lower_wick = np.minimum(data["open"], data["close"]) - data["low"]
        upper_wick = data["high"] - np.maximum(data["open"], data["close"])
        wick_range = (data["high"] - data["low"]).rolling(5, min_periods=5).sum()

        out = pd.DataFrame(index=data.index)
        out["compression"] = (short_atr / long_atr.replace(0, np.nan)).fillna(1.0)
        out["wick_bias"] = (
            (lower_wick - upper_wick).rolling(5, min_periods=5).sum()
            / wick_range.replace(0, np.nan)
        ).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy()
        opens = data["open"].to_numpy()
        compression = indicators["compression"].to_numpy()
        bias = indicators["wick_bias"].to_numpy()
        signals = []
        position = 0
        entry_bar = 0
        entry_price = 0.0
        used_squeeze = False

        for i in range(len(data)):
            if compression[i] >= params.compression_threshold:
                used_squeeze = False

            if position:
                if i == entry_bar:
                    entry_price = float(opens[i])
                if i >= entry_bar and (
                    position * (closes[i] / entry_price - 1.0) >= params.profit_target
                    or i - entry_bar >= 18
                ):
                    position = 0
            elif not used_squeeze and compression[i] < params.compression_threshold:
                if bias[i] > 0.12:
                    position = 1
                elif bias[i] < -0.12:
                    position = -1
                if position:
                    used_squeeze = True
                    entry_bar = i + 1

            signals.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
