from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791230476"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params):
        return 127

    def indicators(self, data, params):
        if not 0.0 < params.fractional_order < 1.0 or params.trail_atr <= 0.0:
            raise ValueError("Require 0 < fractional_order < 1 and trail_atr > 0.")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(64)
        for lag in range(1, 64):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # ponytail: fixed 64-lag fractional filter; extend only if truncation dominates results.
        fractional_close = close * weights[0]
        for lag in range(1, 64):
            fractional_close = fractional_close + close.shift(lag) * weights[lag]
        floor = fractional_close.rolling(63, min_periods=63).min()
        ceiling = fractional_close.rolling(63, min_periods=63).max()
        span = (ceiling - floor).replace(0.0, np.nan)
        position = (fractional_close - floor) / span
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        atr = true_range.rolling(14, min_periods=14).mean()
        shock = close.diff().abs() / atr.shift(1).replace(0.0, np.nan)
        recent_shock = shock.rolling(4, min_periods=4).max()
        ready = position.notna() & position.shift(1).notna() & atr.gt(0) & recent_shock.notna()
        quiet = recent_shock.le(2.5)
        long_entry = ready & quiet & position.ge(0.8) & position.shift(1).lt(0.8)
        short_entry = ready & quiet & position.le(0.2) & position.shift(1).gt(0.2)
        entry = long_entry.astype(int) - short_entry.astype(int)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_floor": floor,
            "fractional_ceiling": ceiling,
            "relative_position": position,
            "true_range": true_range,
            "atr": atr,
            "shock": shock,
            "recent_shock": recent_shock,
            "ready": ready,
            "entry": entry
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
            if not valid:
                side = 0
                continue
            if side:
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                # The specified highest-close drawdown exit applies to both directions.
                if close[i] <= stop or i - entry_bar >= 18:
                    side = 0
            elif entries[i]:
                side = int(entries[i])
                entry_bar = i
                # The first in-trade close is the next bar, after the delayed fill.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams(trail_atr=2.5)
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"close": [100., 105., 110., 108., 107., 107.]}, index=index)
    derived = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 10.], "entry": [1, 0, 0, 0, 0, 0]}, index=index)
    assert strategy.generate_signals(prices, derived, None, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    derived["entry"] = [-1, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(prices, derived, None, params).data["signal"].tolist() == [0, -1, -1, -1, -1, 0]
