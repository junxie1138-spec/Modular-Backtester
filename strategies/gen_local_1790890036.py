from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSpringParams:
    range_fast: int = 3
    range_slow: int = 20
    anchor_bars: int = 4
    reversal_threshold: float = 0.2

    def __post_init__(self):
        if not (1 <= self.range_fast <= self.range_slow and self.anchor_bars >= 2):
            raise ValueError("Require 1 <= range_fast <= range_slow and anchor_bars >= 2")
        if not np.isfinite(self.reversal_threshold) or self.reversal_threshold <= 0:
            raise ValueError("reversal_threshold must be finite and positive")


class GeneratedStrategy(BaseStrategy[RangeSpringParams]):
    strategy_id = "gen_local_1790890036"

    @classmethod
    def params_type(cls):
        return RangeSpringParams

    def warmup_bars(self, params):
        return int(max(params.range_slow, params.anchor_bars))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["fast_range"] = out["range"].rolling(params.range_fast, min_periods=params.range_fast).mean()
        out["slow_range"] = out["range"].rolling(params.range_slow, min_periods=params.range_slow).mean()
        out["anchor"] = out["midpoint"].rolling(params.anchor_bars, min_periods=params.anchor_bars).mean()
        scale = out["slow_range"].where(out["slow_range"] > 0)
        out["range_ratio"] = out["fast_range"] / scale
        # Expansion follows displacement; contraction applies restoring force.
        out["regime"] = np.sign(out["range_ratio"] - 1.0)
        out["displacement"] = (out["midpoint"] - out["anchor"]) / scale
        out["score"] = out["regime"] * out["displacement"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        state = 0
        positions = np.zeros(len(data), dtype=int)
        # Mirrored thresholds: +threshold enters, -threshold reverses to flat.
        # Neutral or unavailable observations retain the current position.
        for i, score in enumerate(indicators["score"].to_numpy(dtype=float)):
            if np.isfinite(score):
                if score >= params.reversal_threshold:
                    state = 1
                elif score <= -params.reversal_threshold:
                    state = 0
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeSpringParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    scores = pd.DataFrame({"score": [np.nan, 0.3, 0.0, np.nan, -0.3, 0.0, 0.3, -0.3]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, scores, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert result.index.equals(index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
