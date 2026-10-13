from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    fractional_order: float = 0.4
    fractional_window: int = 24
    scale_window: int = 32
    shape_bars: int = 5
    history_bars: int = 252
    neighbors: int = 12
    horizon_bars: int = 4
    entry_threshold: float = 0.15
    min_size: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791247347"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.fractional_window + params.scale_window
                   + params.shape_bars + params.history_bars
                   + params.horizon_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.scale_window) < 2:
            raise ValueError("fractional_window and scale_window must be at least two")
        if params.shape_bars < 2 or params.neighbors < 1:
            raise ValueError("Invalid shape or neighbor count")
        if params.horizon_bars < 3 or params.horizon_bars > 5:
            raise ValueError("horizon_bars must be between three and five")
        if params.history_bars < params.neighbors + params.horizon_bars + params.shape_bars:
            raise ValueError("Insufficient analog history")
        if not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("Position sizes must be positive fractions at most one")
        if not np.isfinite(params.entry_threshold):
            raise ValueError("entry_threshold must be finite")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0.0))
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fractional_window):
            weights[k] = weights[k - 1] * (k - 1 - params.fractional_order) / k
        fractional = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional = fractional + weight * out["log_close"].shift(k)
        out["fractional_close"] = fractional
        out["fractional_change"] = fractional.diff()
        out["fractional_scale"] = out["fractional_change"].rolling(
            params.scale_window, min_periods=params.scale_window
        ).std(ddof=0)
        out["scaled_change"] = out["fractional_change"].div(
            out["fractional_scale"].where(out["fractional_scale"] > 1e-12)
        )
        out["log_return"] = out["log_close"].diff()
        out["return_volatility"] = out["log_return"].rolling(
            params.scale_window, min_periods=params.scale_window
        ).std(ddof=0)
        out["horizon_return"] = out["log_close"].diff(params.horizon_bars)
        shape_columns = []
        for lag in range(params.shape_bars):
            name = "shape_" + str(lag)
            out[name] = out["scaled_change"].shift(lag)
            shape_columns.append(name)

        shapes = out[shape_columns].to_numpy(dtype=float)
        returns = out["horizon_return"].to_numpy(dtype=float)
        volatility = out["return_volatility"].to_numpy(dtype=float)
        valid_shapes = np.isfinite(shapes).all(axis=1)
        forecasts = np.full(len(data), np.nan, dtype=float)
        horizon = params.horizon_bars
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not valid_shapes[t]:
                continue
            # Leave a full shape gap after every matured analog outcome.
            candidates = np.arange(max(0, t - params.history_bars),
                                   t - horizon - params.shape_bars + 1)
            outcomes = returns[candidates + horizon]
            scales = volatility[candidates] * np.sqrt(float(horizon))
            valid = (valid_shapes[candidates] & np.isfinite(outcomes)
                     & np.isfinite(scales) & (scales > 1e-12))
            candidates = candidates[valid]
            outcomes = outcomes[valid]
            scales = scales[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.mean((shapes[candidates] - shapes[t]) ** 2, axis=1)
            nearest = np.argpartition(distances, params.neighbors - 1)[:params.neighbors]
            weights = 1.0 / (0.1 + distances[nearest])
            normalized_outcomes = np.clip(outcomes[nearest] / scales[nearest], -3.0, 3.0)
            forecasts[t] = np.average(normalized_outcomes, weights=weights)
        out["analog_forecast"] = forecasts
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        score = indicators["analog_forecast"]
        entry_condition = score.notna() & score.gt(params.entry_threshold)
        df = pd.DataFrame(index=data.index)
        # Maintain the long until the same entry condition flips to false.
        df["signal"] = entry_condition.astype(int)
        conviction = (score - params.entry_threshold).clip(0.0, 1.0).fillna(0.0)
        df["size"] = params.min_size + (params.max_size - params.min_size) * conviction
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AnalogParams()
    strategy = GeneratedStrategy()
    x = np.arange(500, dtype=float)
    close = 100.0 * np.exp(0.0007 * x + 0.009 * np.sin(x / 6.0)
                           + 0.004 * np.sin(x / 17.0))
    data = pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": np.full(len(x), 1000000.0)},
                        index=pd.date_range("2020-01-01", periods=len(x), freq="B"))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = indicators["analog_forecast"].gt(params.entry_threshold).astype(int)
    expected = expected.shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].between(params.min_size, params.max_size).all()
    prefix = data.iloc[:430]
    prefix_indicators = strategy.indicators(prefix, params)
    prefix_result = strategy.generate_signals(prefix, prefix_indicators, None, params).data
    assert np.allclose(indicators.loc[prefix.index, "analog_forecast"],
                       prefix_indicators["analog_forecast"], equal_nan=True)
    assert result.loc[prefix.index].equals(prefix_result)
