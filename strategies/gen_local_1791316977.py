from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 20
    baseline_window: int = 30
    change_z: float = 0.75
    atr_window: int = 14
    trail_atr: float = 2.0
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791316977"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.graph_window + params.baseline_window + 2,
                   params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.graph_window < 3 or params.baseline_window < 2
                or params.atr_window < 1 or params.max_hold < 1
                or not np.isfinite(params.change_z) or params.change_z < 0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        reach = pd.Series(0.0, index=data.index)
        capacity = sum(range(3, params.graph_window + 1))
        # Endpoint horizontal-visibility edges: both endpoints exceed every interior close.
        # ponytail: O(N * graph_window); use a monotonic stack if windows become large.
        for distance in range(3, params.graph_window + 1):
            interior = close.shift(1).rolling(distance - 1, min_periods=distance - 1).max()
            endpoints = np.minimum(close, close.shift(distance))
            reach += endpoints.gt(interior).astype(float) * distance / capacity
        complete = close.rolling(params.graph_window + 1,
                                 min_periods=params.graph_window + 1).count()
        reach = reach.where(complete.eq(params.graph_window + 1))
        baseline = reach.shift(1).rolling(params.baseline_window,
                                         min_periods=params.baseline_window).mean()
        dispersion = reach.shift(1).rolling(params.baseline_window,
                                           min_periods=params.baseline_window).std(ddof=0)
        graph_z = (reach - baseline) / dispersion.clip(lower=1.0 / capacity)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            "reach": reach, "baseline": baseline, "dispersion": dispersion,
            "graph_z": graph_z, "atr": atr, "rising": close.gt(previous)
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        z = indicators["graph_z"].to_numpy(dtype=float)
        rising = indicators["rising"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        streak = 0
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0
                     and np.isfinite(z[i]))
            if not valid:
                active = False
                streak = 0
                age = 0
                high_water = stop = -np.inf
                continue
            candidate = z[i] >= params.change_z and rising[i]
            streak = streak + 1 if candidate else 0
            if active:
                # The entry decision filled at this bar's open, so only now count its close.
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.max_hold:
                    active = False
                    streak = 0
                    high_water = stop = -np.inf
            elif streak >= 2:
                active = True
                age = 0
                high_water = stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100, 101, 102, 103, 105, 104, 101]}, index=index)
    indicators = pd.DataFrame({
        "graph_z": [0, 2, 2, 0, 0, 0, 0],
        "atr": [1, 1, 1, 1, 0.5, 4, 4],
        "rising": [False, True, True, True, True, False, False]
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    # Two confirmations, next-bar entry, and a stop that cannot fall when ATR expands.
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
