from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCurvatureParams:
    lookback: int = 8
    corridor: float = 0.35
    curvature_weight: float = 1.0
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeCurvatureParams]):
    strategy_id = "gen_local-w2_1791211845"

    @classmethod
    def params_type(cls):
        return RangeCurvatureParams

    def warmup_bars(self, params: RangeCurvatureParams) -> int:
        return max(params.lookback + 1, 3)

    def indicators(self, data: pd.DataFrame, params: RangeCurvatureParams) -> pd.DataFrame:
        if params.lookback < 2 or params.corridor <= 0 or params.curvature_weight < 0:
            raise ValueError("Invalid range or corridor parameters")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        daily_range = data["high"] - data["low"]
        out["range"] = daily_range.where(daily_range > 0)
        out["prior_range"] = out["range"].shift(1)
        out["older_range"] = out["range"].shift(2)
        out["reference_range"] = out["prior_range"].rolling(params.lookback, min_periods=params.lookback).median()
        out["curvature"] = (out["range"] - 2 * out["prior_range"] + out["older_range"]) / out["reference_range"]
        out["prior_midpoint"] = (data["high"].shift(1) + data["low"].shift(1)) / 2
        out["displacement"] = (data["close"] - out["prior_midpoint"]) / out["prior_range"]
        out["gain"] = 1 + (params.curvature_weight * out["curvature"]).clip(-0.75, 2.0)
        out["score"] = out["displacement"] * out["gain"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["score"] = out["score"].fillna(0.0)
        out["strength"] = (out["score"].abs() / params.corridor - 1).clip(0, 2) / 2
        out["entry_size"] = params.min_size + (params.max_size - params.min_size) * out["strength"]
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeCurvatureParams) -> SignalFrame:
        score = indicators["score"].fillna(0.0)
        direction = np.select([score > params.corridor, score < -params.corridor], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        held_size = params.min_size
        for i, candidate in enumerate(direction):
            if position == 0 and candidate != 0:
                position = int(candidate)
                held_size = float(indicators["entry_size"].iloc[i])
            elif position != 0 and candidate == -position:
                position = 0
                held_size = params.min_size
            signals[i] = position
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeCurvatureParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"score": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0], "entry_size": [0.6, 0.2, 0.8, 0.8, 0.2, 0.4, 0.7]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, evidence, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"], [0.2, 0.6, 0.6, 0.2, 0.8, 0.8, 0.2])
    index = pd.date_range("2020-01-01", periods=40)
    close = 100 + np.sin(np.arange(40))
    sample = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(sample, params)
    pd.testing.assert_frame_equal(full.iloc[:25], strategy.indicators(sample.iloc[:25], params))
    assert np.isfinite(full.to_numpy()).all()
    assert full["score"].iloc[:params.lookback].eq(0).all()
    assert full["entry_size"].between(params.min_size, params.max_size).all()
    flat = sample.assign(high=100.0, low=100.0, close=100.0, open=100.0)
    assert strategy.indicators(flat, params)["score"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.lookback + 1


if __name__ == "__main__":
    _self_check()
