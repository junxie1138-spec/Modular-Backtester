from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 5
    reference_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    compression_ratio: float = 0.50
    acceleration_threshold: float = 0.002
    trailing_atr: float = 2.0
    max_hold: int = 10

    def __post_init__(self):
        windows = (self.range_window, self.reference_window, self.roc_window, self.atr_window, self.max_hold)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in windows):
            raise ValueError("Lookbacks and max_hold must be positive integers.")
        values = (self.compression_ratio, self.acceleration_threshold, self.trailing_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Thresholds must be finite.")
        if not (0 < self.compression_ratio <= 1) or self.acceleration_threshold < 0 or self.trailing_atr <= 0:
            raise ValueError("Invalid compression, acceleration, or trailing threshold.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791276757"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.reference_window, params.range_window, params.atr_window + 1, params.roc_window + 3))

    def indicators(self, data, params):
        p = params
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["box_high"] = data["high"].rolling(p.range_window, min_periods=p.range_window).max()
        out["box_low"] = data["low"].rolling(p.range_window, min_periods=p.range_window).min()
        out["reference_high"] = data["high"].rolling(p.reference_window, min_periods=p.reference_window).max()
        out["reference_low"] = data["low"].rolling(p.reference_window, min_periods=p.reference_window).min()
        out["box_width"] = out["box_high"] - out["box_low"]
        out["reference_width"] = out["reference_high"] - out["reference_low"]
        out["compression"] = out["box_width"] / out["reference_width"].replace(0, np.nan)
        out["box_midpoint"] = (out["box_high"] + out["box_low"]) / 2
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out["roc"] = close / close.shift(p.roc_window).replace(0, np.nan) - 1
        out["acceleration"] = out["roc"].diff()
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out["ready"] = np.isfinite(out).all(axis=1) & np.isfinite(close) & out["atr"].gt(0)
        out["entry"] = (
            out["ready"]
            & close.gt(out["ma200"])
            & out["compression"].le(p.compression_ratio)
            & out["roc"].lt(0)
            & close.ge(out["box_midpoint"])
            & out["acceleration"].gt(p.acceleration_threshold)
            & out["previous_acceleration"].le(p.acceleration_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        high_water = np.nan
        stop = -np.inf
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if active:
                if not ready[i]:
                    active = False
                else:
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trailing_atr * atr[i])
                    if close[i] <= stop or close[i] <= ma[i] or i - entry_bar + 1 >= params.max_hold:
                        active = False
            elif entry[i] and ready[i]:
                active = True
                entry_bar = i + 1
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    p = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 102., 104., 103., 100., 101.]}, index=index)
    ind = pd.DataFrame({
        "atr": [1., 1., 1., 10., 10., 1.],
        "ma200": 90.,
        "ready": True,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    p.max_hold = 2
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    bars = pd.DataFrame({
        "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.,
    }, index=pd.date_range("2020-01-01", periods=220))
    indicators = strategy.indicators(bars, Params())
    assert indicators.index.equals(bars.index)
    assert not indicators["ready"].iloc[:199].any()
    assert indicators["ready"].iloc[199:].all()
    assert strategy.generate_signals(bars, indicators, None, Params()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
