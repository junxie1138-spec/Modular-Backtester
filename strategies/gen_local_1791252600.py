from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationEchoParams:
    roc_bars: int = 3
    acceleration_bars: int = 2
    correlation_bars: int = 30
    atr_bars: int = 14
    acceleration_min: float = 0.4
    correlation_min: float = 0.15
    echo_min: float = 0.1
    trail_atr: float = 2.5
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.roc_bars, self.acceleration_bars, self.atr_bars, self.hold_bars) < 1 or self.correlation_bars < 3:
            raise ValueError("Lookbacks and holding period must be positive; correlation needs three bars.")
        thresholds = [self.acceleration_min, self.correlation_min, self.echo_min, self.trail_atr]
        if not np.isfinite(thresholds).all() or self.acceleration_min <= 0 or self.echo_min < 0 or self.trail_atr <= 0 or not 0 <= self.correlation_min < 1:
            raise ValueError("Invalid signal or trailing-stop thresholds.")


class GeneratedStrategy(BaseStrategy[AccelerationEchoParams]):
    strategy_id = "gen_local_1791252600"

    @classmethod
    def params_type(cls):
        return AccelerationEchoParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.correlation_bars + 3, params.roc_bars + params.acceleration_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["returns"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["roc_change"] = out["roc"].diff(params.acceleration_bars)
        out["volatility"] = out["returns"].rolling(params.correlation_bars).std(ddof=0).replace(0, np.nan)
        out["acceleration"] = out["roc_change"] / out["volatility"]
        out["lag_two_return"] = out["returns"].shift(2)
        out["autocorrelation"] = out["returns"].rolling(params.correlation_bars).corr(out["lag_two_return"]).clip(-1, 1)
        out["forecast_driver"] = out["returns"].shift(1)
        out["echo"] = out["autocorrelation"] * out["forecast_driver"] / out["volatility"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        values = pd.DataFrame({
            "close": data["close"],
            "atr": indicators["atr"],
            "acceleration": indicators["acceleration"],
            "echo": indicators["echo"],
            "autocorrelation": indicators["autocorrelation"],
        }, index=data.index).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, low_water = np.nan, np.nan
        high_stop, short_stop = -np.inf, np.inf
        for i, (close, atr, acceleration, echo, correlation) in enumerate(values):
            if not np.isfinite(values[i]).all() or close <= 0 or atr <= 0:
                position = 0
                continue
            if position:
                # The position selected yesterday first fills on this bar.
                age += 1
                high_water = close if age == 1 else max(high_water, close)
                low_water = close if age == 1 else min(low_water, close)
                high_stop = max(high_stop, high_water - params.trail_atr * atr)
                short_stop = min(short_stop, low_water + params.trail_atr * atr)
                # High-water exit applies to both sides; shorts also have an adverse-move stop.
                if close <= high_stop or (position == -1 and close >= short_stop) or age >= params.hold_bars:
                    position = 0
            elif correlation >= params.correlation_min:
                if acceleration >= params.acceleration_min and echo > params.echo_min:
                    position = 1
                elif acceleration <= -params.acceleration_min and echo < -params.echo_min:
                    position = -1
                if position:
                    age = 0
                    high_water, low_water = np.nan, np.nan
                    high_stop, short_stop = -np.inf, np.inf
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    close = pd.Series([100, 100, 105, 103, 101, 100, 100, 96, 99], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    test_indicators = pd.DataFrame({
        "atr": 1.0, "autocorrelation": 0.5,
        "acceleration": [1, 0, 0, 0, 0, -1, 0, 0, 0],
        "echo": [1, 0, 0, 0, 0, -1, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, test_indicators, None, AccelerationEchoParams(trail_atr=2.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
