from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryGapParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    normalization_window: int = 40
    atr_window: int = 14
    memory_threshold: float = 0.8
    minimum_gap_atr: float = 0.2
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[MemoryGapParams]):
    strategy_id = "gen_local-w2_1791244964"

    @classmethod
    def params_type(cls):
        return MemoryGapParams

    def warmup_bars(self, params: MemoryGapParams) -> int:
        return int(max(params.fractional_window + params.normalization_window + 2, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: MemoryGapParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0 and params.fractional_window >= 2 and params.normalization_window >= 2 and params.atr_window >= 2 and params.memory_threshold > 0 and params.minimum_gap_atr > 0 and params.stop_atr > 0 and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid fractional-memory, gap, stop, or holding parameters")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        log_close = np.log(close.where(close > 0))
        fractional = log_close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional memory; increase the window for longer memory.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * log_close.shift(lag)
        mean = fractional.rolling(params.normalization_window).mean()
        scale = fractional.rolling(params.normalization_window).std(ddof=0).replace(0.0, np.nan)
        result = pd.DataFrame(index=data.index)
        result["true_range"] = tr
        result["atr"] = atr
        result["log_close"] = log_close
        result["fractional_close"] = fractional
        result["fractional_mean"] = mean
        result["fractional_scale"] = scale
        result["memory_z"] = (fractional - mean) / scale
        result["gap_atr"] = (data["open"] - previous) / atr.shift(1).replace(0.0, np.nan)
        result["close_change"] = close.diff()
        result["memory_direction"] = np.select([result["memory_z"] >= params.memory_threshold, result["memory_z"] <= -params.memory_threshold], [1, -1], default=0).astype(int)
        direction = result["memory_direction"]
        result["confirmed_entry"] = ((direction != 0) & (direction == direction.shift(1)) & (direction * result["gap_atr"].shift(1) <= -params.minimum_gap_atr) & (direction * result["close_change"] > 0) & (atr > 0)).fillna(False)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MemoryGapParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["memory_direction"].fillna(0).to_numpy(dtype=int)
        entries = indicators["confirmed_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Record the opening fill only after its bar has arrived.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = np.isfinite(stop) and np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                invalid = not np.isfinite(stop) or not np.isfinite(closes[i]) or (i == entry_bar and opens[i] <= 0)
                if stopped or expired or invalid:
                    position = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(directions[i])
                entry_bar = i + 1
                entry_atr = atr[i]
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = MemoryGapParams()
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        for trigger_stop in (False, True):
            close = 100.0 + side * np.arange(10, dtype=float)
            if trigger_stop:
                close[4] = 100.0 - side * 3.0
            data = pd.DataFrame({"open": 100.0, "high": 110.0, "low": 90.0, "close": close, "volume": 1000.0}, index=index)
            derived = pd.DataFrame({"atr": 1.0, "memory_direction": side, "confirmed_entry": False}, index=index)
            derived.loc[index[2], "confirmed_entry"] = True
            signal = strategy.generate_signals(data, derived, ctx, params).data["signal"].to_numpy()
            expected = np.zeros(10, dtype=int)
            expected[3:5 if trigger_stop else 7] = side
            assert np.array_equal(signal, expected)
    history = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=100, freq="B"))
    derived = strategy.indicators(history, params)
    assert not derived["confirmed_entry"].any()
    assert (strategy.generate_signals(history, derived, ctx, params).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
