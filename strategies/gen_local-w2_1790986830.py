from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    window: int = 4
    threshold: float = 0.30
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local-w2_1790986830"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return int(2 * max(2, int(params.window)) + 1)

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        window = max(2, int(params.window))
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        opening = data["open"].replace(0.0, np.nan)
        out["gap"] = (opening / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"] / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        area = pd.Series(0.0, index=data.index)
        scale = pd.Series(0.0, index=data.index)
        for lag in range(1, window):
            left = out["gap"].shift(lag) * out["intraday"]
            right = out["intraday"].shift(lag) * out["gap"]
            length = window - lag
            area = area + (left - right).rolling(length, min_periods=length).sum()
            scale = scale + (left.abs() + right.abs()).rolling(length, min_periods=length).sum()
        out["area"] = area
        out["area_scale"] = scale
        out["orientation"] = area.div(scale.replace(0.0, np.nan)).clip(-1.0, 1.0)
        out["prior_orientation"] = out["orientation"].shift(window)
        out["overnight_drift"] = out["gap"].rolling(window, min_periods=window).sum()
        out["rotation"] = (out["orientation"] - out["prior_orientation"]) / 2.0
        out["score"] = (out["rotation"] * np.sign(out["overnight_drift"])).replace([np.inf, -np.inf], np.nan)
        out["ready"] = out[["orientation", "prior_orientation", "overnight_drift", "score"]].notna().all(axis=1)
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RotationParams) -> SignalFrame:
        threshold = float(np.clip(params.threshold, 0.01, 0.99))
        minimum = float(np.clip(params.min_size, 0.01, 1.0))
        score = indicators["score"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        request = np.where(ready & (score >= threshold), 1, np.where(ready & (score <= -threshold), -1, 0))
        strength = np.clip((np.abs(score) - threshold) / (1.0 - threshold), 0.0, 1.0)
        proposed_size = minimum + (1.0 - minimum) * strength
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum, dtype=float)
        position = 0
        held_size = minimum
        for i in range(len(data)):
            direction = int(request[i])
            if position != 0 and direction == -position:
                position = 0
                held_size = minimum
            elif position == 0 and direction != 0:
                position = direction
                held_size = float(proposed_size[i])
            signals[i] = position
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(minimum).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
