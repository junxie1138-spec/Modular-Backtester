from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_bars: int = 5
    correlation_bars: int = 24
    impulse_threshold: float = 0.7
    atr_bars: int = 14
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790892787"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.slope_bars + params.correlation_bars + 3, params.atr_bars + 1))

    def indicators(self, data, params):
        if min(params.slope_bars, params.correlation_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (np.isfinite(params.impulse_threshold) and params.impulse_threshold > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Threshold and ATR multiplier must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].rolling(params.slope_bars).mean()
        out["innovation"] = out["slope"].diff()
        out["lag_innovation"] = out["innovation"].shift(1)
        out["rho"] = out["innovation"].rolling(params.correlation_bars).corr(
            out["lag_innovation"]
        ).shift(1).clip(-0.95, 0.95)
        out["innovation_scale"] = out["innovation"].rolling(
            params.correlation_bars
        ).std().shift(1).replace(0.0, np.nan)
        out["predicted_innovation"] = out["rho"] * out["innovation"]
        out["impulse"] = out["predicted_innovation"] / out["innovation_scale"]
        out["predicted_slope"] = out["slope"] + out["predicted_innovation"]
        out["ma200"] = data["close"].rolling(200).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = np.isfinite(indicators[
            ["impulse", "predicted_slope", "ma200", "atr"]
        ]).all(axis=1) & (indicators["atr"] > 0) & (data["close"] > 0)
        long_entry = valid & (data["close"] > indicators["ma200"]) & (
            indicators["impulse"] > params.impulse_threshold
        ) & (indicators["predicted_slope"] > 0)
        short_entry = valid & (data["close"] < indicators["ma200"]) & (
            indicators["impulse"] < -params.impulse_threshold
        ) & (indicators["predicted_slope"] < 0)
        entries = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, low_water = np.nan, np.nan
        long_stop, short_stop = -np.inf, np.inf
        for i in range(len(data)):
            if position:
                age += 1
                if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                    position = 0
                else:
                    # First held bar is the next bar after the entry decision.
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    low_water = close[i] if np.isnan(low_water) else min(low_water, close[i])
                    long_stop = max(long_stop, high_water - params.trail_atr * atr[i])
                    # Mirror the high-water trailing exit for short positions.
                    short_stop = min(short_stop, low_water + params.trail_atr * atr[i])
                    stopped = (position == 1 and close[i] < long_stop) or (
                        position == -1 and close[i] > short_stop
                    )
                    if stopped or age >= 2:
                        position = 0
                # Exit decisions remain flat; no same-close reentry.
            elif entries[i]:
                position, age = int(entries[i]), 0
                high_water = low_water = np.nan
                long_stop, short_stop = -np.inf, np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        close = np.array([100.0, 100.0 + direction * 4, 100.0 + direction * 2, 100.0])
        data = pd.DataFrame({"close": close}, index=index)
        indicators = pd.DataFrame({
            "impulse": [direction, 0, 0, 0],
            "predicted_slope": direction * 0.01,
            "ma200": 100.0 - direction * 10,
            "atr": 1.0,
        }, index=index)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:3], indicators.iloc[:3], ctx, params).data
        pd.testing.assert_frame_equal(result.iloc[:3], prefix)


if __name__ == "__main__":
    _self_check()
