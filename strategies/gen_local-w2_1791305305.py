from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_bars: int = 63
    shelf_bars: int = 8
    min_depth: float = 0.03
    shelf_width: float = 0.012
    atr_bars: int = 14
    atr_mult: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791305305"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.peak_bars + params.shelf_bars, params.atr_bars + 1))

    def indicators(self, data, params):
        if min(params.peak_bars, params.shelf_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive")
        if not all(np.isfinite(x) and x > 0 for x in (params.min_depth, params.shelf_width, params.atr_mult)):
            raise ValueError("Depth, shelf width and ATR multiplier must be positive")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out["depth"] = 1.0 - close / out["peak"].replace(0.0, np.nan)
        out["shelf_low"] = out["depth"].shift(1).rolling(params.shelf_bars, min_periods=params.shelf_bars).min()
        out["shelf_high"] = out["depth"].shift(1).rolling(params.shelf_bars, min_periods=params.shelf_bars).max()
        out["peak_low"] = out["peak"].rolling(params.shelf_bars + 1, min_periods=params.shelf_bars + 1).min()
        out["peak_high"] = out["peak"].rolling(params.shelf_bars + 1, min_periods=params.shelf_bars + 1).max()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["shelf_width"] = out["shelf_high"] - out["shelf_low"]
        out["ready"] = np.isfinite(out).all(axis=1) & (out["atr"] > 0) & (close > 0)
        shelf = out["ready"] & out["peak_low"].eq(out["peak_high"]) & (out["shelf_low"] >= params.min_depth) & (out["shelf_width"] <= params.shelf_width)
        out["long_entry"] = (shelf & (out["depth"] < out["shelf_low"]) & (close > out["ma200"])).fillna(False)
        out["short_entry"] = (shelf & (out["depth"] > out["shelf_high"]) & (close < out["ma200"])).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water = low_water = stop = np.nan
        values = zip(data["close"].to_numpy(), indicators["atr"].to_numpy(),
                     indicators["ma200"].to_numpy(), indicators["long_entry"].to_numpy(),
                     indicators["short_entry"].to_numpy())
        for i, (close, atr, ma, enter_long, enter_short) in enumerate(values):
            valid = np.isfinite(close) and np.isfinite(atr) and np.isfinite(ma) and atr > 0 and close > 0
            if not valid:
                position, age = 0, 0
                high_water = low_water = stop = np.nan
            elif position:
                age += 1
                high_water = close if np.isnan(high_water) else max(high_water, close)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                if position == 1:
                    candidate = high_water - params.atr_mult * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    exit_trade = close <= stop or close < ma
                else:
                    candidate = low_water + params.atr_mult * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    exit_trade = close >= stop or close > ma
                if exit_trade or age >= params.holding_bars:
                    position, age = 0, 0
                    high_water = low_water = stop = np.nan
            elif enter_long or enter_short:
                position = 1 if enter_long else -1
                age = 0
                # The first in-trade close is on the following execution bar.
                high_water = low_water = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    for direction in (1, -1):
        close = [100.0, 104.0, 103.0, 101.0, 99.0] if direction == 1 else [100.0, 96.0, 97.0, 99.0, 101.0]
        index = pd.date_range("2020-01-01", periods=5)
        data = pd.DataFrame({"close": close}, index=index)
        indicators = pd.DataFrame({
            "atr": [1.0, 1.0, 10.0, 1.0, 1.0],
            "ma200": 90.0 if direction == 1 else 110.0,
            "long_entry": [direction == 1, False, False, False, False],
            "short_entry": [direction == -1, False, False, False, False],
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
