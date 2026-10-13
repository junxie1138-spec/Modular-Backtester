from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalEfficiencyParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    efficiency_window: int = 6
    baseline_window: int = 20
    strength_threshold: float = 0.65
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalEfficiencyParams]):
    strategy_id = "gen_local-w2_1790994129"

    @classmethod
    def params_type(cls) -> type[FractionalEfficiencyParams]:
        return FractionalEfficiencyParams

    def warmup_bars(self, params: FractionalEfficiencyParams) -> int:
        return int(max(200, max(2, params.fractional_window) + max(max(2, params.efficiency_window) + 1, max(2, params.baseline_window)) + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalEfficiencyParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        window = max(2, int(params.fractional_window))
        efficiency_window = max(2, int(params.efficiency_window))
        baseline_window = max(2, int(params.baseline_window))
        order = float(np.clip(params.fractional_order, 0.05, 0.95))
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, window):
            weight *= (lag - 1 - order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        fractional_step = fractional_close.diff()
        progress = fractional_step.rolling(efficiency_window, min_periods=efficiency_window).sum()
        travel = fractional_step.abs().rolling(efficiency_window, min_periods=efficiency_window).sum()
        efficiency = progress.div(travel.where(travel > 0.0)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_step": fractional_step,
            "progress": progress,
            "travel": travel,
            "efficiency": efficiency,
            "previous_efficiency": efficiency.shift(1),
            "fractional_baseline": fractional_close.rolling(baseline_window, min_periods=baseline_window).mean(),
            "ma200": close.rolling(200, min_periods=200).mean(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalEfficiencyParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        eligible = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["fractional_close"] > indicators["fractional_baseline"])
            & (indicators["efficiency"] > params.strength_threshold)
            & (indicators["previous_efficiency"] <= params.strength_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        hold_bars = max(1, int(params.hold_bars))
        signal = np.zeros(len(data), dtype=np.int64)
        exit_at = -1
        for i, can_enter in enumerate(eligible):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif can_enter:
                signal[i] = 1
                exit_at = i + hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
