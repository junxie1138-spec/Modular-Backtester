from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    atr_window: int = 14
    strength_threshold: float = 0.9
    trail_atr: float = 1.75
    holding_bars: int = 4
    min_size: float = 0.2
    max_size: float = 0.95

    def __post_init__(self):
        if self.lookback < 2 or self.atr_window < 1 or not 3 <= self.holding_bars <= 5:
            raise ValueError("Invalid lookback or holding horizon")
        values = [self.strength_threshold, self.trail_atr, self.min_size, self.max_size]
        if not np.isfinite(values).all() or self.strength_threshold <= 0 or self.trail_atr <= 0:
            raise ValueError("Threshold and trailing multiplier must be finite and positive")
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791243241"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(2 * params.lookback, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        peak = close.rolling(params.lookback, min_periods=params.lookback).max()
        trough = close.rolling(params.lookback, min_periods=params.lookback).min()
        drawdown = 1.0 - close / peak
        drawup = close / trough - 1.0
        downside_depth = drawdown.rolling(params.lookback, min_periods=params.lookback).max()
        upside_depth = drawup.rolling(params.lookback, min_periods=params.lookback).max()
        displacement = close / close.shift(params.lookback) - 1.0
        adverse_depth = downside_depth.where(displacement >= 0, upside_depth)
        denominator = adverse_depth + atr / close
        strength = (displacement / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        ready = strength.notna() & atr.notna() & (atr > 0) & close.notna()
        return pd.DataFrame({
            "true_range": true_range,
            "atr": atr,
            "rolling_peak": peak,
            "rolling_trough": trough,
            "drawdown": drawdown,
            "drawup": drawup,
            "downside_depth": downside_depth,
            "upside_depth": upside_depth,
            "displacement": displacement,
            "adverse_depth": adverse_depth,
            "denominator": denominator,
            "strength": strength.fillna(0.0),
            "ready": ready,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        strength = indicators["strength"]
        previous = strength.shift(1).fillna(0.0)
        previous_ready = indicators["ready"].shift(1, fill_value=False)
        long_entry = (strength >= params.strength_threshold) & (previous < params.strength_threshold)
        short_entry = (strength <= -params.strength_threshold) & (previous > -params.strength_threshold)
        entries = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index)
        entries = entries.where(indicators["ready"] & previous_ready, 0).to_numpy(dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = strength.to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        entry_decision = -1
        water = -np.inf
        stop = -np.inf
        entry_size = params.min_size
        for i in range(len(data)):
            if not ready[i] or not np.isfinite(closes[i]) or not np.isfinite(atrs[i]):
                side = 0
                continue
            if side:
                # Start tracking on the first filled bar, not the decision bar.
                # Long: highest close. Short: highest negative close, mirroring the stop.
                signed_close = side * closes[i]
                water = max(water, signed_close)
                stop = max(stop, water - params.trail_atr * atrs[i])
                if signed_close <= stop or i - entry_decision >= params.holding_bars:
                    side = 0
                    continue
            elif entries[i]:
                side = int(entries[i])
                entry_decision = i
                water = -np.inf
                stop = -np.inf
                conviction = float(np.clip(abs(scores[i]) / (2.0 * params.strength_threshold), 0.0, 1.0))
                entry_size = params.min_size + (params.max_size - params.min_size) * conviction
            signals[i] = side
            if side:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=10)
    close = np.array([100, 100, 103, 104, 101, 100, 98, 97, 100, 100], dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "strength": [0, 2, 0, 0, 0, -2, 0, 0, 0, 0],
        "atr": [1, 1, 1, 4, 4, 1, 1, 4, 4, 1],
        "ready": True,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert result["size"].iloc[2] == params.max_size
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    indicators["strength"] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    indicators["atr"] = 1.0
    timed = strategy.generate_signals(flat, indicators, None, params).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert params.min_size < timed["size"].iloc[2] < params.max_size
    warm = strategy.indicators(flat, params)
    assert not warm["ready"].any()
    assert not strategy.generate_signals(flat, warm, None, params).data["signal"].any()
    assert strategy.warmup_bars(params) >= 2 * params.lookback


if __name__ == "__main__":
    _self_check()
