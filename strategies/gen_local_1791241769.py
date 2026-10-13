from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSpringParams:
    memory: int = 8
    fractional_order: float = 0.5
    debt_threshold: float = 0.35
    release_threshold: float = 0.10

    def __post_init__(self):
        if not isinstance(self.memory, int) or not 1 <= self.memory <= 9:
            raise ValueError("memory must be an integer from 1 through 9")
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not all(np.isfinite(x) and x >= 0.0 for x in
                   (self.debt_threshold, self.release_threshold)):
            raise ValueError("thresholds must be finite and nonnegative")


class GeneratedStrategy(BaseStrategy[GapSpringParams]):
    strategy_id = "gen_local_1791241769"

    @classmethod
    def params_type(cls):
        return GapSpringParams

    @staticmethod
    def warmup_bars(params):
        return max(6, params.memory + 1)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["down_gap"] = (-out["gap"]).clip(lower=0.0)
        out["closing_deficit"] = (out["previous_close"] - close).clip(lower=0.0)
        out["unrepaired_gap"] = pd.concat(
            [out["down_gap"], out["closing_deficit"]], axis=1
        ).min(axis=1, skipna=False)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.memory).mean().replace(0.0, np.nan)
        out["gap_debt"] = out["unrepaired_gap"].rolling(params.memory).sum() / out["atr"]
        out["anchor"] = close.shift(5)
        # ponytail: six-tap constant-history approximation; longer memory needs more warmup.
        fractional_close = close - out["anchor"]
        weight = 1.0
        for k in range(1, 6):
            weight *= (k - 1 - params.fractional_order) / k
            fractional_close = fractional_close + weight * (close.shift(k) - out["anchor"])
        out["fractional_close"] = fractional_close
        out["release"] = out["fractional_close"] / out["atr"]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        eligible = (
            (indicators["gap_debt"] > params.debt_threshold)
            & (indicators["release"] > params.release_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # A true condition holds the position; its reversal is the sole exit.
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    close = pd.Series(
        [100, 99, 98, 97, 96, 95, 95.1, 95.3, 95.7, 96.1,
         96.5, 96.9, 97.3, 97.7, 98.1, 98.5, 99, 99.5, 100, 100.5],
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    opening = close.shift(1).fillna(close.iloc[0]) - 0.8
    data = pd.DataFrame({
        "open": opening,
        "high": pd.concat([opening, close], axis=1).max(axis=1) + 0.1,
        "low": pd.concat([opening, close], axis=1).min(axis=1) - 0.1,
        "close": close,
        "volume": 1000.0,
    })
    params = GapSpringParams()
    ind = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    expected = ((ind["gap_debt"] > params.debt_threshold)
                & (ind["release"] > params.release_threshold)).astype(int)
    assert result["signal"].equals(expected.shift(1).fillna(0).astype(int))
    assert result["signal"].max() == 1 and result["signal"].iloc[-1] == 0
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()
    prefix = data.iloc[:12]
    pd.testing.assert_frame_equal(
        ind.iloc[:12], GeneratedStrategy.indicators(prefix, params)
    )
    for memory in (1, 6, 8, 9):
        assert GeneratedStrategy.warmup_bars(GapSpringParams(memory=memory)) <= 10


if __name__ == "__main__":
    _self_check()
