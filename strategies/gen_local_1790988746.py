from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeInnovationParams:
    slope_window: int = 12
    memory_window: int = 30
    percentile_window: int = 126
    entry_percentile: float = 0.85
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[SlopeInnovationParams]):
    strategy_id = "gen_local_1790988746"

    @classmethod
    def params_type(cls) -> type[SlopeInnovationParams]:
        return SlopeInnovationParams

    def warmup_bars(self, params: SlopeInnovationParams) -> int:
        return int(max(2, params.slope_window) + 2 * max(3, params.memory_window) + max(2, params.percentile_window) + 4)

    def indicators(self, data: pd.DataFrame, params: SlopeInnovationParams) -> pd.DataFrame:
        n = max(2, int(params.slope_window))
        memory = max(3, int(params.memory_window))
        history = max(2, int(params.percentile_window))
        percentile = float(np.clip(params.entry_percentile, 0.01, 0.99))
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].astype(float).where(data["close"] > 0))
        out["time"] = np.arange(len(data), dtype=float)
        out["time_log_close"] = out["time"] * out["log_close"]
        out["sum_log_close"] = out["log_close"].rolling(n, min_periods=n).sum()
        out["sum_time"] = out["time"].rolling(n, min_periods=n).sum()
        out["sum_time_log_close"] = out["time_log_close"].rolling(n, min_periods=n).sum()
        denominator = n * (n * n - 1) / 12.0
        out["slope"] = (out["sum_time_log_close"] - out["sum_time"] * out["sum_log_close"] / n) / denominator
        out["expected_slope"] = out["slope"].shift(1).rolling(memory, min_periods=memory).mean()
        out["innovation"] = out["slope"] - out["expected_slope"]
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["persistence"] = out["innovation"].rolling(memory, min_periods=memory).corr(out["lagged_innovation"]).clip(-1.0, 1.0)
        out["propagation"] = out["innovation"] * out["persistence"].clip(lower=0.0)
        out["strength"] = out["propagation"].abs()
        out["entry_threshold"] = out["strength"].shift(1).rolling(history, min_periods=history).quantile(percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SlopeInnovationParams) -> SignalFrame:
        required = ["slope", "innovation", "persistence", "propagation", "entry_threshold"]
        ready = indicators[required].notna().all(axis=1)
        qualifying = ready & (indicators["persistence"] > 0.0) & (indicators["strength"] > indicators["entry_threshold"])
        entries = np.zeros(len(data), dtype=int)
        entries[(qualifying & (indicators["propagation"] > 0.0) & (indicators["slope"] > 0.0)).to_numpy()] = 1
        entries[(qualifying & (indicators["propagation"] < 0.0) & (indicators["slope"] < 0.0)).to_numpy()] = -1
        signals = np.zeros(len(data), dtype=int)
        holding = max(1, int(params.holding_bars))
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i >= exit_bar:
                    position = 0
                signals[i] = position
            elif entry != 0:
                position = int(entry)
                exit_bar = i + holding
                signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
