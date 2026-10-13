from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_bars: int = 30
    reset_bars: int = 3
    entry_z: float = 0.8
    max_z: float = 2.5
    acceleration: float = 0.15
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791318435"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_bars + params.reset_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ma_bars, params.reset_bars, params.atr_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not (0 < params.entry_z < params.max_z and params.acceleration > 0
                and params.trail_atr > 0 and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid thresholds or position sizes.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["dz"] = out["z"].diff()
        out["reset_max"] = out["dz"].shift(1).rolling(params.reset_bars).max()
        out["reset_min"] = out["dz"].shift(1).rolling(params.reset_bars).min()
        out["held_drift"] = close.shift(1) - close.shift(params.reset_bars + 1)
        previous = close.shift(1)
        out["tr"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_bars).mean()
        long_entry = (
            out["z"].between(params.entry_z, params.max_z)
            & (out["dz"] >= params.acceleration)
            & (out["reset_max"] < 0) & (out["held_drift"] >= 0)
            & (out["z"].shift(params.reset_bars + 1) > 0)
        )
        short_entry = (
            out["z"].between(-params.max_z, -params.entry_z)
            & (out["dz"] <= -params.acceleration)
            & (out["reset_min"] > 0) & (out["held_drift"] <= 0)
            & (out["z"].shift(params.reset_bars + 1) < 0)
        )
        out["entry"] = long_entry.astype(int) - short_entry.astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        entry_bar = 0
        extreme = stop = 0.0
        size = params.min_size
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            z = float(indicators["z"].iloc[i])
            if not np.isfinite([close, atr, z]).all() or close <= 0 or atr <= 0:
                side = 0
                continue
            if side:
                if side == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.max_hold:
                    side = 0
                    continue
            else:
                candidate = indicators["entry"].iloc[i]
                if pd.notna(candidate) and candidate in (-1, 1):
                    side = int(candidate)
                    entry_bar = i
                    extreme = close
                    stop = close - side * params.trail_atr * atr
                    strength = np.clip(
                        (abs(z) - params.entry_z) / (params.max_z - params.entry_z), 0, 1
                    )
                    size = float(params.min_size + strength * (params.max_size - params.min_size))
            signals[i] = side
            if side:
                sizes[i] = size
        df["signal"] = signals
        df["size"] = pd.Series(sizes, index=data.index).shift(1).fillna(params.min_size)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(trail_atr=2.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    index = pd.date_range("2020-01-01", periods=5)
    for direction, closes in ((1, [100, 105, 104, 102, 101]),
                              (-1, [100, 95, 96, 98, 99])):
        data = pd.DataFrame({"close": closes}, index=index)
        ind = pd.DataFrame(
            {"atr": [1, 1, 10, 10, 10], "z": [direction * 1.5] * 5,
             "entry": [direction, 0, 0, 0, 0]}, index=index
        )
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].between(params.min_size, params.max_size).all()
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=pd.date_range("2020-01-01", periods=60),
    )
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
