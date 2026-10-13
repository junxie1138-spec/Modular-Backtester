from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTopologyParams:
    window: int = 18
    smoothing: int = 7
    threshold: float = 0.04


class GeneratedStrategy(BaseStrategy[GapTopologyParams]):
    strategy_id = "gen_local_1790876992"

    @classmethod
    def params_type(cls) -> type[GapTopologyParams]:
        return GapTopologyParams

    def warmup_bars(self, params: GapTopologyParams) -> int:
        return int(params.window + params.smoothing)

    @staticmethod
    def _transitivity(windows: np.ndarray) -> np.ndarray:
        count, width = windows.shape
        adjacency = np.zeros((count, width, width), dtype=np.int16)
        # ponytail: dense graphs suit short windows; use sparse edges for large windows.
        for left in range(width - 1):
            barrier = np.full(count, -np.inf)
            for right in range(left + 1, width):
                visible = barrier < np.minimum(windows[:, left], windows[:, right])
                adjacency[:, left, right] = visible
                adjacency[:, right, left] = visible
                barrier = np.maximum(barrier, windows[:, right])
        degrees = adjacency.sum(axis=2)
        wedges = (degrees * (degrees - 1)).sum(axis=1)
        closed = (adjacency * (adjacency @ adjacency)).sum(axis=(1, 2))
        return np.divide(closed, wedges, out=np.zeros(count), where=wedges > 0)

    def indicators(self, data: pd.DataFrame, params: GapTopologyParams) -> pd.DataFrame:
        if params.window < 3 or params.window > 64 or params.smoothing < 1:
            raise ValueError("window must be 3..64 and smoothing must be positive")
        if not np.isfinite(params.threshold) or not 0 < params.threshold < 1:
            raise ValueError("threshold must be finite and between zero and one")
        result = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        ratio = data["open"].where(data["open"] > 0) / previous.where(previous > 0)
        result["gap"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        result["up_closure"] = np.nan
        result["down_closure"] = np.nan
        if len(data) >= params.window:
            windows = np.lib.stride_tricks.sliding_window_view(
                result["gap"].to_numpy(dtype=float), params.window
            )
            valid = np.isfinite(windows).all(axis=1)
            positions = np.flatnonzero(valid) + params.window - 1
            if len(positions):
                result.iloc[positions, result.columns.get_loc("up_closure")] = (
                    self._transitivity(windows[valid])
                )
                result.iloc[positions, result.columns.get_loc("down_closure")] = (
                    self._transitivity(-windows[valid])
                )
        result["closure_balance"] = result["up_closure"] - result["down_closure"]
        result["score"] = result["closure_balance"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTopologyParams,
    ) -> SignalFrame:
        # Mirrored thresholds: neutral topology preserves the current position.
        events = pd.Series(np.nan, index=data.index, dtype=float)
        score = indicators["score"]
        events.loc[score > params.threshold] = 1.0
        events.loc[score < -params.threshold] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    valley = np.array([[3.0, 1.0, 2.0]])
    assert strategy._transitivity(valley)[0] == 1.0
    assert strategy._transitivity(-valley)[0] == 0.0
    sample = pd.DataFrame(index=pd.date_range("2020-01-01", periods=5))
    topology = pd.DataFrame({"score": [np.nan, 0.1, 0.0, -0.1, 0.1]}, index=sample.index)
    signals = strategy.generate_signals(sample, topology, None, GapTopologyParams()).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0]
