from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 5
    pressure_threshold: float = 0.12
    holding_bars: int = 18

    def __post_init__(self):
        if not 2 <= self.rank_window <= 5:
            raise ValueError("rank_window must be between 2 and 5")
        if not 0.0 < self.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be in (0, 1]")
        if self.holding_bars < 1:
            raise ValueError("holding_bars must be positive")


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791252524"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.rank_window + 5)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["body"] = ((data["close"] - data["open"]) / out["range"]).clip(-1.0, 1.0)
        out["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
        out["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
        out["wick_balance"] = ((out["lower_wick"] - out["upper_wick"]) / out["range"]).clip(-1.0, 1.0)
        for feature in ("body", "wick_balance", "range"):
            out[feature + "_rank"] = out[feature].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        for feature in ("body_rank", "wick_balance_rank"):
            out[feature + "_previous"] = out[feature].shift(1)
        features = ("body_rank", "wick_balance_rank", "range_rank", "body_rank_previous", "wick_balance_rank_previous")
        out["weight_sum"] = 0.0
        out["weighted_pressure"] = 0.0
        out["analog_count"] = 0
        for lag in (2, 3, 4):
            distance_column = "distance_" + str(lag)
            weight_column = "weight_" + str(lag)
            outcome_column = "outcome_" + str(lag)
            out[distance_column] = sum((out[feature] - out[feature].shift(lag)).abs() for feature in features) / len(features)
            out[outcome_column] = out["body"].shift(lag - 1)
            out[weight_column] = (1.0 - out[distance_column] / 0.30).clip(lower=0.0).where(out[outcome_column].notna(), 0.0).fillna(0.0)
            out["weight_sum"] += out[weight_column]
            out["weighted_pressure"] += out[weight_column] * out[outcome_column].fillna(0.0)
            out["analog_count"] += (out[weight_column] > 0.0).astype(int)
        out["forecast"] = (out["weighted_pressure"] / out["weight_sum"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        forecast = indicators["forecast"].to_numpy(dtype=float)
        analog_count = indicators["analog_count"].to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if position != 0:
                if i == exit_at:
                    position = 0
                signal[i] = position
                continue
            if analog_count[i] >= 2 and np.isfinite(forecast[i]):
                if forecast[i] >= params.pressure_threshold:
                    position = 1
                elif forecast[i] <= -params.pressure_threshold:
                    position = -1
                if position != 0:
                    exit_at = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
