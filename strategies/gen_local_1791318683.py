from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    acceleration_lag: int = 3
    correlation_bars: int = 60
    min_correlation: float = 0.15
    acceleration_threshold: float = 0.5
    atr_bars: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791318683"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(2 * params.roc_bars + params.correlation_bars,
                       params.roc_bars + params.acceleration_lag + 1,
                       params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["lagged_roc"] = out["roc"].shift(params.roc_bars)
        out["acceleration"] = out["roc"].diff(params.acceleration_lag)
        out["autocorrelation"] = out["roc"].rolling(
            params.correlation_bars, min_periods=params.correlation_bars
        ).corr(out["lagged_roc"])
        out["fractional_atr"] = (out["atr"] / close).where(out["atr"] > 0)
        out["acceleration_score"] = out["acceleration"] / out["fractional_atr"]
        out["continuation_forecast"] = out["autocorrelation"] * out["lagged_roc"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = np.isfinite(indicators[[
            "atr", "autocorrelation", "acceleration_score", "continuation_forecast"
        ]]).all(axis=1) & (indicators["atr"] > 0)
        persistent = ready & (indicators["autocorrelation"] >= params.min_correlation)
        long_entry = persistent & (indicators["acceleration_score"] > params.acceleration_threshold) & (indicators["continuation_forecast"] > 0)
        short_entry = persistent & (indicators["acceleration_score"] < -params.acceleration_threshold) & (indicators["continuation_forecast"] < 0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = 0.0
        stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position = 0
                continue
            if position:
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                # The required high-water drawdown exit applies to both directions.
                if close <= stop or i - entry_bar >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i
                high_water = close
                stop = close - params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100, 103, 105, 103, 101, 100]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 5, 1, 1],
        "autocorrelation": [0.5] * 6,
        "acceleration_score": [1, 0, 0, 0, 0, 0],
        "continuation_forecast": [0.01] * 6
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    indicators["acceleration_score"] *= -1
    indicators["continuation_forecast"] *= -1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0]
    params.hold_bars = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
