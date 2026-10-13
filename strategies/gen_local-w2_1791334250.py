from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DeformationParams:
    displacement_window: int = 4
    baseline_window: int = 12
    threshold: float = 0.30


class GeneratedStrategy(BaseStrategy[DeformationParams]):
    strategy_id = "gen_local-w2_1791334250"

    @classmethod
    def params_type(cls) -> type[DeformationParams]:
        return DeformationParams

    def warmup_bars(self, params: DeformationParams) -> int:
        return int(max(params.displacement_window + 1, params.baseline_window))

    def indicators(self, data: pd.DataFrame, params: DeformationParams) -> pd.DataFrame:
        if params.displacement_window < 1 or params.baseline_window < 1:
            raise ValueError("Lookback windows must be positive")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("Threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["range_width"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["range_center"] = (data["high"] + data["low"]) / 2.0
        out["center_change"] = out["range_center"].diff()
        out["width_change"] = out["range_width"].diff().abs()
        out["displacement"] = out["center_change"].rolling(
            params.displacement_window, min_periods=params.displacement_window
        ).sum()
        out["deformation"] = out["width_change"].rolling(
            params.displacement_window, min_periods=params.displacement_window
        ).sum()
        out["baseline_width"] = out["range_width"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["normalizer"] = out["baseline_width"] + out["deformation"]
        out["plastic_score"] = (
            out["displacement"] / out["normalizer"].where(out["normalizer"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DeformationParams,
    ) -> SignalFrame:
        score = indicators["plastic_score"]
        condition = pd.Series(0, index=data.index, dtype=int)
        condition.loc[score >= params.threshold] = 1
        condition.loc[score <= -params.threshold] = -1
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(condition.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
