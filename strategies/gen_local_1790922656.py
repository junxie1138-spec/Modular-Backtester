from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.65
    fractional_window: int = 32
    score_window: int = 60
    persistence_window: int = 40
    persistence_threshold: float = 0.35
    entry_threshold: float = 1.25
    atr_window: int = 14
    stop_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if min(self.fractional_window, self.score_window, self.persistence_window, self.atr_window) < 2:
            raise ValueError("lookbacks must be at least two")
        if not -1.0 < self.persistence_threshold < 1.0:
            raise ValueError("persistence_threshold must be between minus one and one")
        if not np.isfinite(self.entry_threshold) or self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive and finite")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or self.hold_bars < 1:
            raise ValueError("stop_atr and hold_bars must be positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790922656"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fractional_window + params.score_window - 1,
                       params.fractional_window + params.persistence_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        frac = close.copy()
        weight = 1.0
        # ponytail: finite fractional kernel; increase fractional_window for longer memory.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            frac = frac + weight * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = frac
        out["fractional_mean"] = frac.rolling(params.score_window).mean()
        out["fractional_std"] = frac.rolling(params.score_window).std(ddof=0)
        out["score"] = (frac - out["fractional_mean"]) / out["fractional_std"].replace(0.0, np.nan)
        out["persistence"] = frac.rolling(params.persistence_window).corr(frac.shift(1)).clip(-1.0, 1.0)
        out["ma200"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = np.isfinite(out[["score", "persistence", "ma200", "atr"]]).all(axis=1) & (out["atr"] > 0)
        out["persistent"] = out["persistence"] > params.persistence_threshold
        out["entry_score"] = out["score"].where(out["persistent"], -out["score"])
        long = ready & (close > out["ma200"]) & (out["entry_score"] >= params.entry_threshold)
        short = ready & (close < out["ma200"]) & (out["entry_score"] <= -params.entry_threshold)
        out["long_setup"] = long
        out["short_setup"] = short
        out["entry"] = (long & ~long.shift(1, fill_value=False)).astype(int) - (short & ~short.shift(1, fill_value=False)).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        stop_distance = 0.0
        age = 0
        for i in range(len(data)):
            if position:
                if not np.isfinite(entry_price):
                    # Only read the fill bar's open when that bar is reached.
                    entry_price = opening[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                age += 1
                stopped = (position * (close[i] - entry_price) < -stop_distance)
                if not np.isfinite(close[i]) or stopped or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    continue
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]) and close[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
                age = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    dates = pd.date_range("2020-01-01", periods=5)
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0.0, 10.0, -5.0, -7.0, 0.0])
        bars = pd.DataFrame({"open": 100.0, "close": prices}, index=dates)
        features = pd.DataFrame({"atr": 2.0, "entry": [direction, 0, 0, 0, 0]}, index=dates)
        result = strategy.generate_signals(bars, features, None, Params()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        timed = strategy.generate_signals(bars, features, None, Params(hold_bars=2)).data
        assert timed["signal"].tolist() == [0, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
