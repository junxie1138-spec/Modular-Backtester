from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    shock_atr: float = 0.8
    min_depth: float = 0.012
    trail_atr: float = 1.0
    max_hold: int = 3


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791302061"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = np.maximum.reduce([
            (data["high"] - data["low"]).to_numpy(),
            (data["high"] - previous_close).abs().to_numpy(),
            (data["low"] - previous_close).abs().to_numpy(),
        ])
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200).mean()
        out["atr"] = pd.Series(true_range, index=data.index).rolling(14).mean()
        peak = close.rolling(20).max()
        depth = (peak - close) / peak.replace(0, np.nan)
        out["depth"] = depth
        out["innovation"] = depth - depth.shift(1).rolling(5).mean()
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"]
        atr = indicators["atr"]
        innovation = indicators["innovation"]
        threshold = params.shock_atr * atr / close.replace(0, np.nan)
        entry = (
            (close > indicators["ma200"])
            & (indicators["depth"] >= params.min_depth)
            & (innovation > threshold)
            & (innovation.shift(1) <= threshold.shift(1))
            & (close > data["open"])
        ).fillna(False)

        signals = []
        held = False
        age = 0
        high_water = float("-inf")
        stop = float("-inf")
        prices = close.to_numpy(dtype=float)
        atr_values = atr.to_numpy(dtype=float)
        for i, wants_entry in enumerate(entry.to_numpy()):
            if i < self.warmup_bars(params):
                signals.append(0)
            elif held:
                age += 1
                high_water = max(high_water, prices[i])
                stop = max(stop, high_water - params.trail_atr * atr_values[i])
                if prices[i] <= stop or prices[i] > prices[i - 1] or age >= params.max_hold:
                    held = False
                    signals.append(0)
                else:
                    signals.append(1)
            elif wants_entry:
                held = True
                age = 0
                high_water = float("-inf")
                stop = float("-inf")
                signals.append(1)
            else:
                signals.append(0)

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
