from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueParams:
    depth_threshold: float = 0.75
    overload: float = 5.5
    quiet_limit: float = 1.5


class GeneratedStrategy(BaseStrategy[QueueParams]):
    strategy_id = "gen_local_1791000954"

    @classmethod
    def params_type(cls) -> type[QueueParams]:
        return QueueParams

    def warmup_bars(self, params: QueueParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: QueueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["peak"] = close.rolling(3, min_periods=3).max()
        out["trough"] = close.rolling(3, min_periods=3).min()
        out["capacity"] = (data["high"] - data["low"]).clip(lower=0.0).rolling(3, min_periods=3).mean().replace(0.0, np.nan)
        out["drawdown_depth"] = ((out["peak"] - close) / out["capacity"]).replace([np.inf, -np.inf], np.nan)
        out["drawup_depth"] = ((close - out["trough"]) / out["capacity"]).replace([np.inf, -np.inf], np.nan)
        out["down_arrival"] = (out["drawdown_depth"] - params.depth_threshold).clip(lower=0.0, upper=1.0)
        out["up_arrival"] = (out["drawup_depth"] - params.depth_threshold).clip(lower=0.0, upper=1.0)
        out["down_queue"] = out["down_arrival"].rolling(8, min_periods=8).sum()
        out["up_queue"] = out["up_arrival"].rolling(8, min_periods=8).sum()
        out["ready"] = out[["down_queue", "up_queue"]].notna().all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QueueParams) -> SignalFrame:
        ready = indicators["ready"].astype(bool)
        long_entry = ready & (indicators["up_queue"] >= params.overload) & (indicators["down_queue"] <= params.quiet_limit)
        short_entry = ready & (indicators["down_queue"] >= params.overload) & (indicators["up_queue"] <= params.quiet_limit)
        candidate = np.select([long_entry & ~short_entry, short_entry & ~long_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
