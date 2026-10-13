from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 5
    volatility_bars: int = 20
    shape_gap: int = 3
    memory_bars: int = 504
    neighbors: int = 5
    hold_bars: int = 18
    max_distance: float = 1.0
    min_edge: float = 0.01
    min_win_fraction: float = 0.6


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790904251"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.memory_bars + 2 * params.shape_gap + max(2 * params.roc_bars + 1, params.volatility_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        opening = data["open"].astype(float)
        result = pd.DataFrame(index=data.index)
        result["daily_return"] = close.pct_change(fill_method=None)
        result["volatility"] = result["daily_return"].rolling(params.volatility_bars, min_periods=params.volatility_bars).std(ddof=0)
        result["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        result["acceleration"] = result["roc"].diff(params.roc_bars)
        scale = result["volatility"].replace(0.0, np.nan) * np.sqrt(params.roc_bars)
        result["shape_now"] = result["acceleration"] / scale
        result["shape_middle"] = result["shape_now"].shift(params.shape_gap)
        result["shape_old"] = result["shape_now"].shift(2 * params.shape_gap)
        features = result[["shape_now", "shape_middle", "shape_old"]].replace([np.inf, -np.inf], np.nan)
        first_lag = params.hold_bars + 2 * params.shape_gap + 2
        lags = list(range(first_lag, params.memory_bars + 1, max(1, params.hold_bars)))
        result["analog_edge"] = np.nan
        result["analog_win_fraction"] = np.nan
        result["analog_distance"] = np.nan
        if not lags:
            return result
        distances = np.full((len(data), len(lags)), np.inf, dtype=float)
        outcomes = np.full_like(distances, np.nan)
        current = features.to_numpy(dtype=float)
        for column, lag in enumerate(lags):
            historical = features.shift(lag).to_numpy(dtype=float)
            outcome = (opening.shift(lag - params.hold_bars - 1) / opening.shift(lag - 1).replace(0.0, np.nan) - 1.0).to_numpy(dtype=float)
            distance = np.sqrt(np.mean((current - historical) ** 2, axis=1))
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[valid, column] = distance[valid]
            outcomes[valid, column] = outcome[valid]
        count = min(max(1, params.neighbors), len(lags))
        nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
        matched_distance = np.take_along_axis(distances, nearest, axis=1)
        matched_outcome = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(matched_distance).all(axis=1) & np.isfinite(matched_outcome).all(axis=1)
        result.loc[complete, "analog_edge"] = matched_outcome[complete].mean(axis=1)
        result.loc[complete, "analog_win_fraction"] = (matched_outcome[complete] > 0.0).mean(axis=1)
        result.loc[complete, "analog_distance"] = matched_distance[complete].max(axis=1)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        eligible = (
            indicators["shape_now"].gt(0.0)
            & indicators["analog_edge"].ge(params.min_edge)
            & indicators["analog_win_fraction"].ge(params.min_win_fraction)
            & indicators["analog_distance"].le(params.max_distance)
        ).fillna(False)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for bar, entry in enumerate(confirmed.to_numpy(dtype=bool)):
            if remaining > 0:
                remaining -= 1
                if remaining > 0:
                    signal[bar] = 1
            elif entry:
                remaining = max(1, int(params.hold_bars))
                signal[bar] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
