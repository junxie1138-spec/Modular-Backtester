from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 40
    smooth_bars: int = 3
    degree_threshold: float = 0.25
    displacement_threshold: float = 0.9


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790873683"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_bars + params.smooth_bars + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_bars < 3 or params.smooth_bars < 1:
            raise ValueError("graph_bars >= 3 and smooth_bars >= 1 required")
        if not (np.isfinite(params.degree_threshold) and 0 < params.degree_threshold < 1):
            raise ValueError("degree_threshold must be between zero and one")
        if not (np.isfinite(params.displacement_threshold) and params.displacement_threshold > 0):
            raise ValueError("displacement_threshold must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_close = np.log(close.where((close > 0) & np.isfinite(close)))
        out["log_close"] = log_close
        n = len(data)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        max_slope = np.full(n, -np.inf)
        min_slope = np.full(n, np.inf)
        # Endpoint visibility is a record-slope test; vectorize across all dates.
        for lag in range(1, params.graph_bars):
            slope = ((log_close.shift(lag) - log_close) / lag).to_numpy()
            finite = np.isfinite(slope)
            upper += finite & (slope > max_slope)
            lower += finite & (slope < min_slope)
            max_slope = np.maximum(max_slope, np.where(finite, slope, -np.inf))
            min_slope = np.minimum(min_slope, np.where(finite, slope, np.inf))
        ready = log_close.rolling(params.graph_bars).count().eq(params.graph_bars)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(ready)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(ready)
        out["degree_balance"] = (out["lower_degree"] - out["upper_degree"]) / (params.graph_bars - 1)
        out["balance"] = out["degree_balance"].rolling(params.smooth_bars).mean()
        mean = log_close.rolling(params.graph_bars).mean()
        std = log_close.rolling(params.graph_bars).std(ddof=0).replace(0.0, np.nan)
        out["log_mean"] = mean
        out["log_std"] = std
        out["displacement"] = (log_close - mean) / std
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        long_setup = (
            (indicators["balance"] >= params.degree_threshold)
            & (indicators["displacement"] <= -params.displacement_threshold)
        )
        short_setup = (
            (indicators["balance"] <= -params.degree_threshold)
            & (indicators["displacement"] >= params.displacement_threshold)
        )
        intent = np.where(long_setup, 1, np.where(short_setup, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, side in enumerate(intent):
            if position != 0:
                if side == -position:
                    position = 0
            elif side != 0 and side == previous:
                position = int(side)
            positions[i] = position
            previous = int(side)
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {"balance": [0, 0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, 0.5],
         "displacement": [0, -2, -2, 0, 2, 2, 0, -2, -2]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    small = VisibilityParams(graph_bars=3, smooth_bars=1)
    valley = pd.DataFrame({"close": np.exp([2.0, 1.5, 0.0])})
    topology = strategy.indicators(valley, small)
    assert topology["lower_degree"].iloc[-1] == 2.0
    assert topology["upper_degree"].iloc[-1] == 1.0
    flat = strategy.indicators(data, VisibilityParams(graph_bars=3, smooth_bars=1))
    assert flat["displacement"].isna().all()


if __name__ == "__main__":
    _self_check()
