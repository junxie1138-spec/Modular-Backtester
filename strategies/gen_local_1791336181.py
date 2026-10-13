from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_bars: int = 20
    atr_bars: int = 14
    shape_bars: int = 5
    library_bars: int = 126
    percentile_bars: int = 126
    match_quantile: float = 0.10
    edge_quantile: float = 0.80
    displacement_quantile: float = 0.35
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791336181"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params):
        return int(max(params.atr_bars + 1, params.ma_bars + params.library_bars
                       + params.hold_bars + 2 * params.shape_bars
                       + params.percentile_bars - 1))

    def indicators(self, data, params):
        p = params
        if (min(p.ma_bars, p.atr_bars, p.shape_bars, p.library_bars,
                p.percentile_bars) < 2 or not 3 <= p.hold_bars <= 5
                or p.stop_atr <= 0
                or not all(0 < q < 1 for q in
                           (p.match_quantile, p.edge_quantile,
                            p.displacement_quantile))):
            raise ValueError("Invalid analog strategy parameters")
        close = data["close"]
        mean = close.rolling(p.ma_bars).mean()
        deviation = close.rolling(p.ma_bars).std(ddof=0).replace(0, np.nan)
        z = ((close - mean) / deviation).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"],
                                (data["high"] - previous).abs(),
                                (data["low"] - previous).abs()], axis=1).max(axis=1)
        values = {"ma": mean, "deviation": deviation, "z": z,
                  "true_range": true_range,
                  "atr": true_range.rolling(p.atr_bars).mean()}
        shape = [z.shift(j) for j in range(p.shape_bars)]
        for j, feature in enumerate(shape):
            values[f"shape_{j}"] = feature
        distances, outcomes = [], []
        # Embargo keeps analog shapes AND their outcomes outside today's shape.
        first_lag = p.hold_bars + p.shape_bars
        for lag in range(first_lag, first_lag + p.library_bars):
            distance = np.sqrt(sum((feature - feature.shift(lag)) ** 2
                                   for feature in shape) / p.shape_bars)
            outcome = (close.shift(lag - p.hold_bars)
                       / close.shift(lag).replace(0, np.nan) - 1)
            outcome = outcome.replace([np.inf, -np.inf], np.nan)
            values[f"distance_{lag}"] = distance
            values[f"outcome_{lag}"] = outcome
            distances.append(distance)
            outcomes.append(outcome)
        distance_frame = pd.concat(distances, axis=1)
        outcome_frame = pd.concat(outcomes, axis=1)
        ready = distance_frame.notna().all(axis=1) & outcome_frame.notna().all(axis=1)
        cutoff = distance_frame.quantile(p.match_quantile, axis=1)
        neighbors = distance_frame.le(cutoff, axis=0).to_numpy()
        selected = outcome_frame.where(neighbors)
        forecast = selected.mean(axis=1).where(ready)
        baseline = outcome_frame.mean(axis=1).where(ready)
        edge = forecast - baseline
        edge_cutoff = edge.shift(1).rolling(p.percentile_bars).quantile(p.edge_quantile)
        z_cutoff = z.shift(1).rolling(p.percentile_bars).quantile(p.displacement_quantile)
        values.update({"distance_cutoff": cutoff, "neighbor_count": selected.count(axis=1),
                       "forecast": forecast, "baseline": baseline, "edge": edge,
                       "edge_cutoff": edge_cutoff, "z_cutoff": z_cutoff})
        values["entry"] = ((forecast > 0) & (edge > edge_cutoff)
                           & (z <= z_cutoff) & (values["atr"] > 0)).fillna(False)
        return pd.DataFrame(values, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.stop_atr * atr[i])
                if (not valid or close[i] <= stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and valid:
                active = True
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams(hold_bars=5)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 105., 104., 102., 101., 100.]}, index=index)
    indicators = pd.DataFrame({"atr": [1., 1., 1., 4., 4., 1., 1.],
                               "entry": [True, False, False, False, False, False, False]},
                              index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators["atr"] = 10.0
    result = strategy.generate_signals(data, indicators, ctx, AnalogParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
