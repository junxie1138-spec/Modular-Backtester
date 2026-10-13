from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAbsorptionParams:
    atr_window: int = 14
    min_gap_atr: float = 0.6
    min_first_fill: float = 0.35
    trail_atr: float = 0.7


class GeneratedStrategy(BaseStrategy[GapAbsorptionParams]):
    strategy_id = "gen_local_1790857142"

    @classmethod
    def params_type(cls):
        return GapAbsorptionParams

    def warmup_bars(self, params):
        return int(params.atr_window + 1)

    def indicators(self, data, params):
        if (params.atr_window < 1 or params.min_gap_atr <= 0
                or not 0 < params.min_first_fill < 1 or params.trail_atr <= 0):
            raise ValueError("Invalid gap absorption parameters")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        out["prior_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_atr"] = out["gap"] / out["prior_atr"]
        out["gap_fill"] = (
            (data["close"] - data["open"]) / (-out["gap"]).replace(0.0, np.nan)
        )
        out["first_rejection"] = (
            (out["gap_atr"] <= -params.min_gap_atr)
            & (out["gap_fill"] >= params.min_first_fill)
            & (out["gap_fill"] < 1.0)
        )
        out["confirmed"] = (
            out["first_rejection"].shift(1, fill_value=False)
            & (out["gap"] < 0.0)
            & (out["gap"].abs() < out["gap"].shift(1).abs())
            & (out["gap_fill"] >= 1.0)
            & (data["close"] < out["prior_close"].shift(1))
            & np.isfinite(out["atr"])
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                age = 0
                high_water = np.nan
                stop = -np.inf
                continue
            if active:
                # The entry decision was yesterday; this close is now in-trade.
                age += 1
                high_water = close[i] if age == 1 else max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= 2:
                    active = False
                    age = 0
                    high_water = np.nan
                    stop = -np.inf
            elif confirmed[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            signal[i] = int(active)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    data = pd.DataFrame({
        "open": [100, 100, 100, 98, 98.5, 100, 99, 99],
        "close": [100, 100, 100, 99, 99.5, 100.4, 98.5, 99],
        "high": [101, 101, 101, 100, 100, 101, 100, 100],
        "low": [99, 99, 99, 97.5, 98, 99.5, 98, 98],
        "volume": [1000] * 8,
    }, index=pd.date_range("2020-01-01", periods=8))
    strategy = GeneratedStrategy()
    params = GapAbsorptionParams(atr_window=2)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
    for end in range(1, len(data) + 1):
        prefix = data.iloc[:end]
        actual = strategy.generate_signals(
            prefix, strategy.indicators(prefix, params), ctx, params
        ).data
        assert actual["signal"].equals(result["signal"].iloc[:end])


if __name__ == "__main__":
    _self_check()
