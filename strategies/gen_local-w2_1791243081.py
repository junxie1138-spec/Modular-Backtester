from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 12
    degree_threshold: float = 0.25

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 3:
            raise ValueError("lookback must be an integer >= 3")
        if not 0.0 < self.degree_threshold <= 1.0:
            raise ValueError("degree_threshold must be in (0, 1]")


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791243081"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params):
        return params.lookback

    def indicators(self, data, params):
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        current = close.to_numpy(dtype=float)
        upper = np.zeros(len(data), dtype=float)
        lower = np.zeros(len(data), dtype=float)
        ceiling = np.full(len(data), -np.inf)
        floor = np.full(len(data), np.inf)
        # An edge exists when every intervening close lies below both endpoints;
        # the reflected graph reverses this inequality.
        for lag in range(1, params.lookback):
            past = close.shift(lag).to_numpy(dtype=float)
            finite = np.isfinite(current) & np.isfinite(past)
            upper += finite & (np.minimum(current, past) > ceiling)
            lower += finite & (np.maximum(current, past) < floor)
            ceiling = np.maximum(ceiling, past)
            floor = np.minimum(floor, past)
        out = pd.DataFrame(index=data.index)
        out["window_low"] = close.rolling(params.lookback).min()
        out["window_high"] = close.rolling(params.lookback).max()
        out["valid"] = close.rolling(params.lookback).count().eq(params.lookback)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(out["valid"])
        out["lower_degree"] = pd.Series(lower, index=data.index).where(out["valid"])
        span = (out["window_high"] - out["window_low"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["window_low"]) / span
        out["degree_balance"] = (out["lower_degree"] - out["upper_degree"]) / (params.lookback - 1)
        out["entry"] = np.select(
            [
                out["valid"] & out["relative_position"].le(0.25)
                & out["degree_balance"].ge(params.degree_threshold),
                out["valid"] & out["relative_position"].ge(0.75)
                & out["degree_balance"].le(-params.degree_threshold),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        for i, entry in enumerate(indicators["entry"].fillna(0).to_numpy()):
            if position:
                age += 1
                if age == 2:
                    position = 0
                    age = 0
            elif entry:
                position = int(entry)
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VisibilityParams(lookback=4)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0]}, index=index)
    graph = GeneratedStrategy.indicators(None, data, params)
    assert graph["entry"].tolist() == [0, 0, 0, 0, 0, 0, -1]
    assert graph["upper_degree"].iloc[-1] == 3
    assert graph["lower_degree"].iloc[-1] == 1
    falling = data.copy()
    falling["close"] = 8.0 - data["close"]
    assert GeneratedStrategy.indicators(None, falling, params)["entry"].iloc[-1] == 1
    entries = pd.DataFrame({"entry": [1, -1, -1, -1, 1, 1, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    prefix = GeneratedStrategy.indicators(None, data.iloc[:5], params)
    pd.testing.assert_frame_equal(graph.iloc[:5], prefix)


if __name__ == "__main__":
    _self_check()
