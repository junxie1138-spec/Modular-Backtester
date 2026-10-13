from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.4
    fractional_window: int = 20
    compression_window: int = 4
    baseline_window: int = 24
    compression_ratio: float = 0.75
    pressure_window: int = 3
    pressure_threshold: float = 0.15
    atr_window: int = 14
    stop_atr: float = 1.0
    hold_bars: int = 2

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must lie between zero and one")
        if self.fractional_window < 2 or min(self.compression_window, self.baseline_window, self.pressure_window, self.atr_window) < 1:
            raise ValueError("lookbacks must be positive; fractional_window must be at least two")
        if self.compression_window >= self.baseline_window:
            raise ValueError("compression_window must be shorter than baseline_window")
        if not 0.0 < self.compression_ratio < 1.0 or not 0.0 <= self.pressure_threshold < 1.0 or not self.stop_atr > 0.0:
            raise ValueError("invalid compression, pressure, or stop threshold")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791345076"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fractional_window + 2, params.baseline_window + 1, params.compression_window + 1, params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        # ponytail: fixed-length fractional filter; enlarge fractional_window for a longer memory.
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional = fractional + close.shift(lag) * weights[lag]
        out["fractional_close"] = fractional
        out["fractional_step"] = fractional.diff()
        out["compression"] = (out["true_range"].rolling(params.compression_window, min_periods=params.compression_window).mean() / out["true_range"].rolling(params.baseline_window, min_periods=params.baseline_window).mean().replace(0.0, np.nan)).shift(1)
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["close_location"] = (2.0 * close - data["high"] - data["low"]) / span
        out["pressure"] = out["close_location"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        step = out["fractional_step"]
        out["fractional_turn"] = np.select([(step > 0.0) & (step.shift(1) <= 0.0), (step < 0.0) & (step.shift(1) >= 0.0)], [1, -1], default=0)
        out["pressure_direction"] = np.select([out["pressure"] > params.pressure_threshold, out["pressure"] < -params.pressure_threshold], [1, -1], default=0)
        ready = out[["atr", "compression", "fractional_step", "pressure"]].notna().all(axis=1) & (out["atr"] > 0.0)
        agreement = (out["fractional_turn"] == out["pressure_direction"]) & (out["fractional_turn"] != 0)
        out["entry"] = out["fractional_turn"].where(ready & agreement & (out["compression"] <= params.compression_ratio), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        age = 0
        stop = np.nan
        for i in range(self.warmup_bars(params), len(data)):
            if side:
                if age == 0:
                    # The preceding close requested entry; today's open is its fill.
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                age += 1
                if not np.isfinite(stop) or not np.isfinite(closes[i]) or side * (closes[i] - stop) < 0.0 or age >= params.hold_bars:
                    side = 0
                    age = 0
            elif entries[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]):
                side = int(entries[i])
                age = 0
            desired[i] = side
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    w = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=w + 11, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    computed = strategy.indicators(data, params)
    assert computed.index.equals(index)
    assert not np.isinf(computed.to_numpy(dtype=float)).any()
    controlled = pd.DataFrame({"atr": 1.0, "entry": 0}, index=index)
    controlled.loc[index[[w, w + 6]], "entry"] = 1
    controlled.loc[index[w + 3], "entry"] = -1
    data.loc[index[w + 1], "close"] = 98.0
    data.loc[index[w + 4], "close"] = 102.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=w)
    result = strategy.generate_signals(data, controlled, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[[w + 1, w + 7, w + 8]] = 1
    expected[w + 4] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
