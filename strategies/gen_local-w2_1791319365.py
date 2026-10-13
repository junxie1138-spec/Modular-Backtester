from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    baseline_window: int = 10
    compression_window: int = 3
    min_streak: int = 2
    compression_ratio: float = 1.0
    edge_threshold: float = 0.65
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local-w2_1791319365"

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(params.baseline_window + 1,
                       params.compression_window,
                       2 * params.min_streak))

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if min(params.baseline_window, params.compression_window, params.min_streak) < 1:
            raise ValueError("Window lengths and min_streak must be positive")
        if not np.isfinite(params.compression_ratio) or params.compression_ratio <= 0:
            raise ValueError("compression_ratio must be finite and positive")
        if not 0.5 < params.edge_threshold < 1.0:
            raise ValueError("edge_threshold must lie between 0.5 and 1")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")

        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"])
        out["relative_range"] = (out["range"] / data["close"].where(
            data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["baseline_range"] = out["relative_range"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window).mean()
        out["recent_range"] = out["relative_range"].rolling(
            params.compression_window, min_periods=params.compression_window).mean()
        out["range_ratio"] = (out["recent_range"] / out["baseline_range"].where(
            out["baseline_range"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["close_location"] = ((data["close"] - data["low"]) /
            out["range"]).replace([np.inf, -np.inf], np.nan).clip(0, 1)

        upper = out["close_location"].ge(params.edge_threshold).fillna(False)
        lower = out["close_location"].le(1 - params.edge_threshold).fillna(False)
        out["upper_streak"] = upper.astype(int).groupby(
            (~upper).cumsum()).cumsum().clip(upper=2 * params.min_streak)
        out["lower_streak"] = lower.astype(int).groupby(
            (~lower).cumsum()).cumsum().clip(upper=2 * params.min_streak)
        out["compressed"] = out["range_ratio"].le(
            params.compression_ratio).fillna(False)
        out["setup"] = np.select(
            [out["compressed"] & out["lower_streak"].ge(params.min_streak),
             out["compressed"] & out["upper_streak"].ge(params.min_streak)],
            [1, -1], default=0).astype(int)
        out["streak_strength"] = (out[["upper_streak", "lower_streak"]].max(
            axis=1) / (2.0 * params.min_streak)).clip(0, 1)
        out["compression_strength"] = (1 - out["range_ratio"] /
            (2 * params.compression_ratio)).clip(0, 1).fillna(0)
        out["entry_size"] = params.min_size + (
            params.max_size - params.min_size) * out["streak_strength"] * out[
            "compression_strength"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        held_size = params.min_size
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(params.min_size).to_numpy(
            dtype=float)

        for i in range(len(data)):
            direction = int(setups[i])
            if position != 0 and direction == -position:
                position = 0
                held_size = params.min_size
            elif position == 0 and direction != 0:
                position = direction
                held_size = float(entry_sizes[i])
            elif position != 0 and direction == position:
                held_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = PressureParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"setup": [0, 1, 0, -1, -1, 0, 1],
                               "entry_size": [0.25, 0.6, 0.3, 0.8, 0.7, 0.4, 0.9]},
                              index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert np.allclose(result.data["size"], [0.25, 0.25, 0.6, 0.6, 0.25, 0.7, 0.7])
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived["setup"].eq(0).all()
    assert derived["entry_size"].gt(0).all()
