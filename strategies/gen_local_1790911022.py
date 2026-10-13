from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    range_window: int = 12
    roc_window: int = 3
    acceleration_pct: float = 0.15
    position_threshold: float = 0.65
    atr_window: int = 14
    breakeven_pct: float = 0.8
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790911022"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.range_window + 2, params.roc_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.range_window, params.roc_window, params.atr_window) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 < params.position_threshold < 1.0:
            raise ValueError("position_threshold must be between zero and one")
        if params.trail_atr <= 0.0 or params.breakeven_pct <= 0.0:
            raise ValueError("Stop parameters must be positive")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["roc"] = 100.0 * (close / close.shift(params.roc_window).replace(0.0, np.nan) - 1.0)
        out["acceleration"] = out["roc"].diff()
        out["range_high"] = data["high"].rolling(params.range_window, min_periods=params.range_window).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_window, min_periods=params.range_window).min().shift(1)
        out["range_width"] = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = (close - out["range_low"]) / out["range_width"]
        out["previous_position"] = out["position"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        required = indicators[["roc", "acceleration", "position", "previous_position", "atr"]]
        ready = np.isfinite(required).all(axis=1) & (indicators["atr"] > 0.0)
        acceleration_agrees = (indicators["roc"] > 0.0) & (indicators["acceleration"] >= params.acceleration_pct)
        position_agrees = (indicators["position"] >= params.position_threshold) & (indicators["previous_position"] < params.position_threshold)
        entries = (ready & acceleration_agrees & position_agrees).to_numpy(dtype=bool)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid_price = np.isfinite(prices[i]).all() and min(prices[i]) > 0.0
            if pending:
                pending = False
                if not valid_price or i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    continue
                # Yesterday's decision fills at today's open.
                active = True
                armed = False
                age = 0
                entry = opening
                peak = opening
                stop = entry - params.trail_atr * atr[i - 1]
            if active:
                age += 1
                # Test the stop established before this session; new ratchets
                # become effective after today's close. Exits fill next open.
                if not valid_price or low <= stop or age >= params.hold_bars:
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct / 100.0):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                signals[i] = 1
            elif valid_price and entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [100.0, 102.0, 103.0, 102.0, 102.0, 102.0, 102.0, 102.0],
        "low": [99.0, 99.5, 101.0, 101.25, 99.0, 99.0, 99.0, 99.0],
        "close": [100.0, 101.8, 102.8, 101.6, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    params = StrategyParams()
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = pd.DataFrame({
        "roc": [1.0] * 8,
        "acceleration": [0.2] * 8,
        "position": [0.7] * 8,
        "previous_position": [0.5] + [0.7] * 7,
        "atr": [1.0] * 8,
    }, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    indicators["acceleration"] = 0.0
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert not strategy.generate_signals(data, calculated, ctx, params).data["signal"].any()
