from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 10
    reaction_bars: int = 4
    percentile_window: int = 63
    percentile: float = 0.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790921009"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.graph_window + params.reaction_bars + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_window < 3 or params.reaction_bars < 1 or params.percentile_window < 2 or not 0 < params.percentile < 1:
            raise ValueError("Invalid visibility or percentile parameters")
        x = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame({"log_close": x}, index=data.index)
        complete = x.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        minimum = maximum = x - x.shift(1)
        out["edge_1"] = minimum.notna().astype(float).where(complete)
        # An edge exists if every intervening close lies strictly on one side
        # of its chord: natural visibility on price or inverted price.
        for lag in range(2, params.graph_window):
            slope = (x - x.shift(lag)) / lag
            visible = ((slope < minimum) | (slope > maximum)) & slope.notna()
            out[f"edge_{lag}"] = visible.astype(float).where(complete)
            minimum = np.minimum(minimum, slope)
            maximum = np.maximum(maximum, slope)
        changed = pd.Series(0.0, index=data.index)
        union = pd.Series(0.0, index=data.index)
        valid = complete & complete.shift(1, fill_value=False)
        # Align identical historical anchors across consecutive endpoints.
        for lag in range(2, params.graph_window):
            current = out[f"edge_{lag}"]
            previous = out[f"edge_{lag - 1}"].shift(1)
            changed += current.ne(previous).astype(float)
            union += (current.eq(1) | previous.eq(1)).astype(float)
        out["edge_turnover"] = (changed / union.where(union > 0, 1.0)).where(valid)
        out["rewiring"] = out["edge_turnover"].rolling(params.reaction_bars, min_periods=params.reaction_bars).mean()
        out["threshold"] = out["rewiring"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.percentile)
        out["drift"] = x.diff(params.reaction_bars)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        active = indicators["rewiring"].gt(indicators["threshold"]) & indicators["threshold"].notna()
        direction = np.sign(indicators["drift"].fillna(0)).astype(int)
        entries = direction.where(active, 0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"rewiring": [1, 0, 1, 1, 0, 1, 0], "threshold": 0.5, "drift": [1, 0, -1, -1, 0, 1, 0]}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
