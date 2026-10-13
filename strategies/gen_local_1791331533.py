from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    slope_window: int = 7
    vol_window: int = 20
    shape_length: int = 5
    library_bars: int = 180
    neighbors: int = 9
    horizon: int = 8
    min_edge: float = 0.2
    stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791331533"

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.slope_window + 2, params.vol_window + 1)
                   + 2 * params.shape_length + params.horizon
                   + params.library_bars)

    def indicators(self, data, params):
        if (params.slope_window < 3 or params.vol_window < 2
                or params.shape_length < 2 or params.horizon < 1
                or not 1 <= params.neighbors <= params.library_bars
                or params.min_edge <= 0 or params.stop_atr <= 0
                or params.max_hold < 1):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.vol_window).std().replace(0, np.nan)
        x = np.arange(params.slope_window, dtype=float)
        x -= x.mean()
        denominator = float(np.dot(x, x))
        out["slope"] = sum(
            x[params.slope_window - 1 - lag] * out["log_close"].shift(lag)
            for lag in range(params.slope_window)
        ) / denominator
        out["predicted_slope"] = 2 * out["slope"].shift(1) - out["slope"].shift(2)
        out["innovation"] = (out["slope"] - out["predicted_slope"]) / out["volatility"]
        out["slope_state"] = out["slope"] / out["volatility"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.vol_window).mean()
        features = [out["slope_state"]]
        for lag in range(params.shape_length):
            name = "innovation_lag_" + str(lag)
            out[name] = out["innovation"].shift(lag)
            features.append(out[name])
        distances, outcomes = [], []
        # Each outcome ends before the current fingerprint begins.
        first_lag = params.horizon + params.shape_length
        for lag in range(first_lag, first_lag + params.library_bars):
            distance = sum((feature - feature.shift(lag)) ** 2 for feature in features)
            distances.append((distance / len(features)).to_numpy())
            outcome = (out["log_close"].shift(lag - params.horizon)
                       - out["log_close"].shift(lag))
            outcomes.append((outcome / (out["volatility"].shift(lag)
                                       * np.sqrt(params.horizon))).to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        valid = np.isfinite(distance_matrix) & np.isfinite(outcome_matrix)
        distance_matrix = np.where(valid, distance_matrix, np.inf)
        nearest = np.argpartition(distance_matrix, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distance = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcome_matrix, nearest, axis=1)
        weights = 1.0 / (0.001 + selected_distance)
        selected_outcome = np.where(np.isfinite(selected_outcome), selected_outcome, 0.0)
        total_weight = weights.sum(axis=1)
        safe_weight = np.where(total_weight > 0, total_weight, np.nan)
        ready = np.isfinite(selected_distance).sum(axis=1) == params.neighbors
        out["forecast"] = np.where(ready, (weights * selected_outcome).sum(axis=1) / safe_weight, np.nan)
        out["agreement"] = np.where(ready, (weights * (selected_outcome > 0)).sum(axis=1) / safe_weight, np.nan)
        out["strength"] = ((out["forecast"] / (2 * params.min_edge))
                           * out["agreement"]).clip(0.2, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({"signal": 0, "size": 0.2}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        agreement = indicators["agreement"].to_numpy(dtype=float)
        innovation = indicators["innovation"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        active = False
        entry_bar = 0
        high_water = -np.inf
        stop = -np.inf
        entry_size = 0.2
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.stop_atr * atr[i])
                    if close[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                        active = False
                if active:
                    signals[i] = 1
                    sizes[i] = entry_size
                continue
            if (valid and i >= self.warmup_bars(params) - 1
                    and np.isfinite(forecast[i]) and forecast[i] >= params.min_edge
                    and np.isfinite(agreement[i]) and agreement[i] >= 0.6
                    and np.isfinite(innovation[i]) and innovation[i] > 0
                    and np.isfinite(strength[i])):
                active = True
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
                entry_size = float(np.clip(strength[i], 0.2, 0.95))
                signals[i] = 1
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams()
    padding = strategy.warmup_bars(params) - 1
    closes = [100.0] * padding + [100.0, 100.0, 104.0, 102.0, 99.0, 99.0]
    index = pd.date_range("2020-01-01", periods=len(closes))
    data = pd.DataFrame({"close": closes}, index=index)
    ind = pd.DataFrame({
        "atr": 1.0, "forecast": np.nan, "agreement": 1.0,
        "innovation": 1.0, "strength": 0.8
    }, index=index)
    ind.iloc[padding, ind.columns.get_loc("forecast")] = 1.0
    ind.iloc[padding + 3:, ind.columns.get_loc("atr")] = 10.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[padding:].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].iloc[padding + 1] == 0.8
    flat = pd.DataFrame({"close": [100.0] * (padding + 14)},
                        index=pd.date_range("2020-01-01", periods=padding + 14))
    flat_ind = ind.reindex(flat.index).fillna({
        "atr": 1.0, "agreement": 1.0, "innovation": 1.0, "strength": 0.8
    })
    flat_ind["atr"] = 1.0
    result = strategy.generate_signals(flat, flat_ind, None, params).data
    assert result["signal"].iloc[padding:padding + 12].tolist() == [0] + [1] * 10 + [0]
    flat_ind["forecast"] = np.nan
    assert not strategy.generate_signals(flat, flat_ind, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
