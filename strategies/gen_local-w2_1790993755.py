from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeQueueParams:
    slope_window: int = 5
    memory_window: int = 24
    queue_window: int = 4
    capacity: float = 0.7
    min_autocorrelation: float = 0.15


class GeneratedStrategy(BaseStrategy[SlopeQueueParams]):
    strategy_id = "gen_local-w2_1790993755"

    @classmethod
    def params_type(cls) -> type[SlopeQueueParams]:
        return SlopeQueueParams

    def warmup_bars(self, params: SlopeQueueParams) -> int:
        return int(max(3, params.slope_window) + max(3, params.memory_window) + max(1, params.queue_window) + 2)

    def indicators(self, data: pd.DataFrame, params: SlopeQueueParams) -> pd.DataFrame:
        n = max(3, int(params.slope_window))
        memory = max(3, int(params.memory_window))
        queue = max(1, int(params.queue_window))
        log_close = np.log(data["close"].astype(float).where(data["close"] > 0))
        slope = pd.Series(0.0, index=data.index)
        for lag in range(n):
            weight = (n - 1) / 2.0 - lag
            slope = slope + weight * log_close.shift(lag)
        slope = slope / (n * (n * n - 1) / 12.0)
        innovation = slope.diff()
        lagged_innovation = innovation.shift(1)
        innovation_scale = innovation.rolling(memory, min_periods=memory).std(ddof=0)
        autocorrelation = innovation.rolling(memory, min_periods=memory).corr(lagged_innovation).clip(-1.0, 1.0)
        standardized = innovation / innovation_scale.where(innovation_scale > 1e-12)
        backlog = standardized.rolling(queue, min_periods=queue).sum() / np.sqrt(float(queue))
        score = backlog * autocorrelation.clip(lower=0.0)
        out = pd.DataFrame({
            "log_close": log_close,
            "slope": slope,
            "innovation": innovation,
            "lagged_innovation": lagged_innovation,
            "innovation_scale": innovation_scale,
            "autocorrelation": autocorrelation,
            "standardized_innovation": standardized,
            "backlog": backlog,
            "score": score,
        }, index=data.index)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out.notna().all(axis=1).astype(int)
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SlopeQueueParams) -> SignalFrame:
        eligible = (indicators["ready"] == 1) & (indicators["autocorrelation"] >= max(0.0, params.min_autocorrelation))
        capacity = max(1e-6, float(params.capacity))
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[eligible & (indicators["score"] >= capacity)] = 1
        entry.loc[eligible & (indicators["score"] <= -capacity)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, direction in enumerate(entry.to_numpy(dtype=int)):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # The identical opposite entry condition is the only exit.
                # Emit a flat bar before accepting a subsequent entry.
                position = 0
            positions[bar] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
