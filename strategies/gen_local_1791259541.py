from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    trend_span: int = 5
    innovation_span: int = 12
    vol_window: int = 20
    embedding_step: int = 3
    forecast_horizon: int = 7
    archive_bars: int = 252
    neighbors: int = 9
    entry_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791259541"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(
            max(params.vol_window + 1, params.trend_span + params.innovation_span + 1)
            + 2 * params.embedding_step
            + params.forecast_horizon
            + params.archive_bars
        )

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0)
        out["latent_slope"] = out["return"].ewm(
            span=params.trend_span, adjust=False, min_periods=params.trend_span
        ).mean()
        out["expected_slope"] = out["latent_slope"].shift(1).ewm(
            span=params.innovation_span,
            adjust=False,
            min_periods=params.innovation_span,
        ).mean()
        out["innovation"] = (
            out["latent_slope"] - out["expected_slope"]
        ) / out["volatility"].where(out["volatility"] > 0)
        out["innovation_lag1"] = out["innovation"].shift(params.embedding_step)
        out["innovation_lag2"] = out["innovation"].shift(2 * params.embedding_step)
        features = out[["innovation", "innovation_lag1", "innovation_lag2"]]
        current = features.to_numpy(dtype=float)
        distances = np.full((len(data), params.archive_bars), np.inf)
        outcomes = np.zeros_like(distances)
        for column, lag in enumerate(range(
            params.forecast_horizon + 1,
            params.forecast_horizon + params.archive_bars + 1,
        )):
            historical = features.shift(lag).to_numpy(dtype=float)
            direct = np.sum((current - historical) ** 2, axis=1)
            reflected = np.sum((current + historical) ** 2, axis=1)
            orientation = np.where(direct <= reflected, 1.0, -1.0)
            realized = (
                out["log_close"].shift(lag - params.forecast_horizon)
                - out["log_close"].shift(lag)
            ) / (
                out["volatility"].shift(lag).where(out["volatility"].shift(lag) > 0)
                * np.sqrt(params.forecast_horizon)
            )
            realized_values = realized.to_numpy(dtype=float)
            distance = np.minimum(direct, reflected)
            valid = np.isfinite(distance) & np.isfinite(realized_values)
            distances[:, column] = np.where(valid, distance, np.inf)
            outcomes[:, column] = np.where(valid, orientation * realized_values, 0.0)
        k = min(params.neighbors, params.archive_bars)
        selected = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distance = np.take_along_axis(distances, selected, axis=1)
        nearest_outcome = np.take_along_axis(outcomes, selected, axis=1)
        weights = np.where(np.isfinite(nearest_distance), 1.0 / (1.0 + nearest_distance), 0.0)
        weight_sum = weights.sum(axis=1)
        forecast = np.divide(
            (weights * nearest_outcome).sum(axis=1),
            weight_sum,
            out=np.zeros(len(data), dtype=float),
            where=weight_sum > 0,
        )
        out["analog_count"] = np.isfinite(nearest_distance).sum(axis=1)
        out["analog_distance"] = np.divide(
            np.where(np.isfinite(nearest_distance), nearest_distance, 0.0).sum(axis=1),
            out["analog_count"].to_numpy(),
            out=np.full(len(data), np.nan),
            where=out["analog_count"].to_numpy() > 0,
        )
        reliable = (out["analog_count"].to_numpy() == k) & (out["analog_distance"].to_numpy() <= 1.5)
        out["forecast"] = np.where(reliable & np.isfinite(forecast), forecast, 0.0)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("forecast")] = 0.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"].fillna(0.0).to_numpy(dtype=float)
        entry = np.where(
            forecast > params.entry_threshold,
            1,
            np.where(forecast < -params.entry_threshold, -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=400)
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.008 * np.sin(np.arange(400) / 4.0)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index) and np.isfinite(derived["forecast"]).all()
    sample = data.iloc[:6]
    forecasts = pd.DataFrame({"forecast": [0.5, 0.1, -0.5, -0.1, 0.5, 0.1]}, index=sample.index)
    result = strategy.generate_signals(sample, forecasts, None, params).data
    mirrored = strategy.generate_signals(sample, -forecasts, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert np.array_equal(mirrored["signal"].to_numpy(), -result["signal"].to_numpy())
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
