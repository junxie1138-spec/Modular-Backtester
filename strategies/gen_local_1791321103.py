from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history_bars: int = 504
    neighbors: int = 12
    horizon: int = 8
    energy_window: int = 8
    forecast_smooth: int = 3
    entry_edge: float = 0.001


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791321103"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        feature_bars = 16 + params.energy_window - 1
        last_lag = params.horizon + feature_bars + params.history_bars - 1
        return int(max(200, feature_bars + last_lag + params.forecast_smooth - 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.history_bars < params.neighbors or params.neighbors < 1
                or params.horizon < 1 or params.energy_window < 1
                or params.forecast_smooth < 1 or not np.isfinite(params.entry_edge)):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        energy_columns = []
        for half in (2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = (mean - mean.shift(half)) * np.sqrt(half / 2.0)
            column = f"energy_{half}"
            out[column] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            energy_columns.append(column)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 1e-16)
        features = []
        for half in (2, 4, 8):
            column = f"share_{half}"
            out[column] = out[f"energy_{half}"] / denominator
            features.append(column)
        out["coarse_direction"] = np.tanh(out["haar_8"] / np.sqrt(denominator))
        features.append("coarse_direction")
        shape = out[features].replace([np.inf, -np.inf], np.nan)
        # Completed analog outcomes end before the current descriptor begins.
        first_lag = params.horizon + 16 + params.energy_window - 1
        distances = np.full((len(data), params.history_bars), np.inf)
        outcomes = np.full_like(distances, np.nan)
        current = shape.to_numpy()
        for column, lag in enumerate(range(first_lag, first_lag + params.history_bars)):
            past = shape.shift(lag).to_numpy()
            outcome = (close.shift(lag - params.horizon) / close.shift(lag) - 1.0).to_numpy()
            valid = (np.isfinite(current).all(axis=1)
                     & np.isfinite(past).all(axis=1) & np.isfinite(outcome))
            distances[valid, column] = np.square(current[valid] - past[valid]).sum(axis=1)
            outcomes[valid, column] = outcome[valid]
        nearest = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distance = np.take_along_axis(distances, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcomes, nearest, axis=1)
        valid_neighbors = np.isfinite(selected_distance) & np.isfinite(selected_outcome)
        count = valid_neighbors.sum(axis=1)
        out["neighbor_count"] = count
        total = np.where(valid_neighbors, selected_outcome, 0.0).sum(axis=1)
        out["analog_forecast"] = np.where(
            count == params.neighbors, total / np.maximum(count, 1), np.nan
        )
        out["forecast"] = out["analog_forecast"].rolling(
            params.forecast_smooth, min_periods=params.forecast_smooth
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        ready = np.arange(len(data)) >= self.warmup_bars(params) - 1
        # The full entry predicate defines the state; its reversal is the only exit.
        entry = (ready & (data["close"] > indicators["ma200"])
                 & (indicators["forecast"] > params.entry_edge)
                 & (indicators["neighbor_count"] == params.neighbors))
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.fillna(False).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = AnalogParams(history_bars=40, neighbors=4)
    strategy = object.__new__(GeneratedStrategy)
    n = 360
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.02 * np.sin(t / 7.0))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:300], params)
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)
    signals = strategy.generate_signals(data, full, None, params).data
    assert signals["signal"].isin([0, 1]).all()
    assert (signals["size"] > 0).all()
    assert not signals["signal"].iloc[:strategy.warmup_bars(params)].any()
    altered = full.copy()
    altered["ma200"] = data["close"] - 1.0
    altered["forecast"] = params.entry_edge + 0.01
    altered["neighbor_count"] = params.neighbors
    altered.loc[data.index[250], "forecast"] = -1.0
    check = strategy.generate_signals(data, altered, None, params).data["signal"]
    assert check.iloc[250] == 1 and check.iloc[251] == 0 and check.iloc[252] == 1
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_signals = strategy.generate_signals(
        flat, strategy.indicators(flat, params), None, params
    ).data
    assert not flat_signals["signal"].any()


if __name__ == "__main__":
    _self_check()
