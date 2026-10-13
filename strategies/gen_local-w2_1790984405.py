from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 20
    calibration_window: int = 60
    smoothing_window: int = 18
    entry_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1790984405"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(params.graph_window + params.calibration_window + params.smoothing_window + 1)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if min(params.graph_window, params.smoothing_window) < 2 or params.calibration_window < 3:
            raise ValueError("Graph and smoothing windows must be >= 2; calibration must be >= 3.")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("entry_threshold must be finite and positive.")
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        amplitude = returns.abs()
        degree = pd.Series(0.0, index=data.index)
        neighbor_sum = pd.Series(0.0, index=data.index)
        interior_max = pd.Series(-np.inf, index=data.index)
        for lag in range(1, params.graph_window + 1):
            past = returns.shift(lag)
            past_amplitude = amplitude.shift(lag)
            visible = (
                returns.notna() & past.notna()
                & (interior_max < np.minimum(amplitude, past_amplitude))
            )
            degree += visible.astype(float)
            neighbor_sum += past.where(visible, 0.0)
            interior_max = pd.Series(
                np.maximum(interior_max.to_numpy(), past_amplitude.to_numpy()),
                index=data.index,
            )
        complete = returns.rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        degree = degree.where(complete)
        neighbor_return = (neighbor_sum / degree.replace(0.0, np.nan)).where(complete)
        autocorrelation = returns.rolling(params.calibration_window).corr(neighbor_return).clip(-1.0, 1.0)
        volatility = returns.rolling(params.calibration_window).std().replace(0.0, np.nan)
        forecast = (autocorrelation * neighbor_return / volatility).replace([np.inf, -np.inf], np.nan)
        score = forecast.rolling(params.smoothing_window).mean()
        return pd.DataFrame({
            "return": returns,
            "amplitude": amplitude,
            "visibility_degree": degree,
            "visible_neighbor_return": neighbor_return,
            "edge_autocorrelation": autocorrelation,
            "volatility": volatility,
            "forecast": forecast,
            "score": score,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        score = indicators["score"]
        condition = np.where(
            score > params.entry_threshold, 1,
            np.where(score < -params.entry_threshold, -1, 0),
        )
        signal = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, direction in enumerate(condition):
            if position != 0 and direction == -position:
                position = 0
            elif position == 0 and direction != 0 and direction == previous:
                position = int(direction)
            signal[i] = position
            previous = int(direction)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    scores = pd.DataFrame(
        {"score": [np.nan, 0.2, 0.2, 0.0, np.nan, -0.2, -0.2, -0.2, 0.2, 0.2]},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, scores, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
