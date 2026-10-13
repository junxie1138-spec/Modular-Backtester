from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    lookback: int = 6
    memory: int = 4
    threshold: float = 0.25

    def __post_init__(self):
        if not (2 <= self.lookback and 1 <= self.memory):
            raise ValueError("Windows must be positive, with lookback >= 2.")
        if self.lookback + self.memory - 1 > 10:
            raise ValueError("Combined warmup must not exceed 10 bars.")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must lie between zero and one.")


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791325598"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(params.lookback + params.memory - 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["peak"] = close.rolling(params.lookback, min_periods=params.lookback).max()
        out["trough"] = close.rolling(params.lookback, min_periods=params.lookback).min()
        out["drawdown"] = np.log(out["peak"] / close).clip(lower=0.0)
        out["drawup"] = np.log(close / out["trough"]).clip(lower=0.0)
        out["down_depth"] = out["drawdown"].rolling(params.memory, min_periods=params.memory).mean()
        out["up_depth"] = out["drawup"].rolling(params.memory, min_periods=params.memory).mean()
        out["total_depth"] = out["down_depth"] + out["up_depth"]
        out["ready"] = out["down_depth"].notna() & out["up_depth"].notna()
        out["score"] = ((out["up_depth"] - out["down_depth"]) /
                        out["total_depth"].replace(0.0, np.nan)).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (score, ready) in enumerate(zip(indicators["score"], indicators["ready"])):
            direction = 0
            if ready and np.isfinite(score):
                if score >= params.threshold:
                    direction = 1
                elif score <= -params.threshold:
                    direction = -1
            if position == 0:
                position = direction
            elif direction == -position:
                # An opposite qualifying condition exits; reentry waits until another bar.
                position = 0
            positions[i] = position
        df["signal"] = positions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = DrawdownParams()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=12),
    )
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert not derived["ready"].iloc[:8].any()
    assert derived["ready"].iloc[8:].all()
    assert (strategy.generate_signals(data, derived, None, params).data["signal"] == 0).all()
    sample = data.iloc[:8]
    derived = pd.DataFrame(
        {"score": [0.6, 0.0, -0.6, -0.6, 0.0, 0.6, 0.6, 0.0], "ready": True},
        index=sample.index,
    )
    result = strategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
