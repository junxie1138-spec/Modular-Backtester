from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 4
    match_distance: float = 0.35
    forecast_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790990418"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return min(4, max(2, int(params.rank_window))) + 6

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        window = min(4, max(2, int(params.rank_window)))
        tolerance = max(0.01, float(params.match_distance))
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        body = ((data["close"] - data["open"]) / span).replace([np.inf, -np.inf], np.nan)
        location = ((data["close"] - data["low"]) / span).replace([np.inf, -np.inf], np.nan)
        out["range"] = span
        out["body"] = body
        out["location"] = location
        out["body_rank"] = body.rolling(window, min_periods=window).rank(pct=True)
        out["location_rank"] = location.rolling(window, min_periods=window).rank(pct=True)
        out["previous_body_rank"] = out["body_rank"].shift(1)
        out["previous_location_rank"] = out["location_rank"].shift(1)
        out["close_change"] = data["close"].diff()
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        ready = out[["body_rank", "location_rank", "previous_body_rank", "previous_location_rank"]].notna().all(axis=1)
        features = ["body_rank", "location_rank", "previous_body_rank", "previous_location_rank"]
        for lag in range(2, 6):
            distance = pd.Series(0.0, index=data.index)
            valid = ready.copy()
            for feature in features:
                historical = out[feature].shift(lag)
                distance = distance + (out[feature] - historical).abs()
                valid = valid & historical.notna()
            distance = distance / len(features)
            # Each analog ends at t-lag; its next-day outcome is already known.
            outcome = (out["close_change"].shift(lag - 1) / span.shift(lag)).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
            valid = valid & outcome.notna()
            weight = (1.0 - distance / tolerance).clip(lower=0.0).where(valid, 0.0).fillna(0.0)
            out[f"distance_{lag}"] = distance
            out[f"outcome_{lag}"] = outcome
            out[f"weight_{lag}"] = weight
            numerator = numerator + weight * outcome.fillna(0.0)
            denominator = denominator + weight
            ready = ready & valid
        out["weighted_outcome"] = numerator
        out["total_weight"] = denominator
        out["ready"] = ready
        out["forecast"] = (numerator / denominator.where(denominator > 0)).where(ready).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        threshold = max(0.01, float(params.forecast_threshold))
        forecast = indicators["forecast"]
        desired = np.select([forecast.ge(threshold), forecast.le(-threshold)], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # An opposite qualifying entry condition closes the position.
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
