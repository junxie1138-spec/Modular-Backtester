from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySpanParams:
    graph_window: int = 8
    reversal_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[VisibilitySpanParams]):
    strategy_id = "gen_local-w2_1791297937"

    @classmethod
    def params_type(cls):
        return VisibilitySpanParams

    def warmup_bars(self, params: VisibilitySpanParams) -> int:
        return params.graph_window + 1

    def indicators(self, data: pd.DataFrame, params: VisibilitySpanParams) -> pd.DataFrame:
        if params.graph_window < 3 or not 0 < params.reversal_threshold < 2:
            raise ValueError("Invalid strategy parameters")
        close = np.log(data["close"].astype(float).where(data["close"] > 0))
        low_slope = pd.Series(np.inf, index=data.index)
        high_slope = pd.Series(-np.inf, index=data.index)
        votes = pd.Series(0.0, index=data.index)
        mass = pd.Series(0.0, index=data.index)
        for lag in range(1, params.graph_window):
            slope = (close - close.shift(lag)) / lag
            visible = slope.notna() & ((slope < low_slope) | (slope > high_slope))
            votes += lag * np.sign(slope.fillna(0.0)) * visible
            mass += lag * visible
            low_slope = np.minimum(low_slope, slope)
            high_slope = np.maximum(high_slope, slope)
        complete = close.rolling(params.graph_window).count().eq(params.graph_window)
        score = (votes / mass.where(mass > 0)).where(complete)
        pressure = score.diff()
        ready = score.notna() & pressure.notna()
        return pd.DataFrame({"score": score.fillna(0.0), "pressure": pressure.fillna(0.0), "ready": ready}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilitySpanParams) -> SignalFrame:
        pressure = indicators["pressure"]
        score = indicators["score"]
        ready = indicators["ready"]
        long_entry = (ready & (score > 0) & (pressure > params.reversal_threshold)).to_numpy()
        short_entry = (ready & (score < 0) & (pressure < -params.reversal_threshold)).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        state = 0
        for i in range(len(data)):
            if (state == 1 and short_entry[i]) or (state == -1 and long_entry[i]):
                state = 0
            elif state == 0:
                if long_entry[i]:
                    state = 1
                elif short_entry[i]:
                    state = -1
            raw[i] = state
        size = (pressure.abs() / (2 * params.reversal_threshold)).clip(lower=0.25, upper=1.0)
        df = pd.DataFrame({"signal": raw, "size": size.shift(1).fillna(0.25)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"score": [0.0, 0.8, 0.8, -0.3, -0.8, 0.3], "pressure": [0.0, 0.8, 0.0, -1.1, -0.5, 1.1], "ready": [False, True, True, True, True, True]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, VisibilitySpanParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
