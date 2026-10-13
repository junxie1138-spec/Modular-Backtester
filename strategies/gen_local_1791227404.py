from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 80
    entry_percentile: float = 0.80
    atr_window: int = 14
    stop_atr: float = 1.6
    hold_bars: int = 9


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791227404"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.rank_window + 1, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        previous = data["close"].shift(1)
        gap = ((previous - data["open"]) / previous.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        recovery = ((data["close"] - data["open"]) / (previous - data["open"]).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["gap"] = gap.fillna(0.0)
        out["recovery"] = recovery.fillna(0.0)
        out["threshold"] = gap.rolling(params.rank_window, min_periods=params.rank_window).quantile(params.entry_percentile).fillna(0.0)
        out["high_threshold"] = gap.rolling(params.rank_window, min_periods=params.rank_window).quantile(0.95).fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        entry_price = None
        entry_atr = 0.0
        position_size = 1.0
        held = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if active:
                if entry_price is None:
                    entry_price = float(data["open"].iloc[i])
                held += 1
                if close < entry_price - params.stop_atr * entry_atr or held >= params.hold_bars:
                    active = False
                    entry_price = None
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif i >= self.warmup_bars(params):
                row = indicators.iloc[i]
                gap = float(row["gap"])
                threshold = float(row["threshold"])
                if gap > 0 and gap >= threshold and 0.35 <= row["recovery"] <= 0.90 and row["atr"] > 0:
                    strength = np.clip((gap - threshold) / max(float(row["high_threshold"]) - threshold, 1e-12), 0.0, 1.0)
                    position_size = float(0.35 + 0.60 * strength)
                    entry_atr = float(row["atr"])
                    entry_price = None
                    held = 0
                    active = True
                    signals[i] = 1
                    sizes[i] = position_size
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(sizes, index=data.index).shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
