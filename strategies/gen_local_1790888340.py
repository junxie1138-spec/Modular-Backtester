from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_window: int = 12
    drift_bars: int = 7
    clustering_threshold: float = 0.32


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790888340"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.graph_window + 2, params.drift_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.drift_bars < 1:
            raise ValueError("graph_window must be >= 3 and drift_bars >= 1")
        if not np.isfinite(params.clustering_threshold) or not 0 <= params.clustering_threshold <= 1:
            raise ValueError("clustering_threshold must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        w = params.graph_window
        windows = np.column_stack([
            close.shift(w - 1 - i).to_numpy() for i in range(w)
        ])
        # ponytail: dense small-window graphs; use sparse graphs if windows grow substantially.
        adjacency = np.zeros((len(data), w, w), dtype=float)
        for i in range(w - 1):
            for j in range(i + 1, w):
                if j == i + 1:
                    visible = np.ones(len(data), dtype=bool)
                else:
                    visible = (
                        np.max(windows[:, i + 1:j], axis=1)
                        < np.minimum(windows[:, i], windows[:, j])
                    )
                adjacency[:, i, j] = visible
                adjacency[:, j, i] = visible
        degree = adjacency.sum(axis=2)
        ordered_wedges = (degree * (degree - 1)).sum(axis=1)
        closed_walks = np.einsum(
            "nij,nji->n", adjacency @ adjacency, adjacency
        )
        transitivity = np.divide(
            closed_walks,
            ordered_wedges,
            out=np.zeros(len(data), dtype=float),
            where=ordered_wedges > 0,
        )
        transitivity[~np.isfinite(windows).all(axis=1)] = np.nan
        out["transitivity"] = transitivity
        out["clustering"] = out["transitivity"].rolling(3, min_periods=3).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["drift"] = close.pct_change(params.drift_bars, fill_method=None)
        out["valid_close"] = close
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[
            ["clustering", "ma200", "drift", "valid_close"]
        ].to_numpy()).all(axis=1)
        follow = indicators["clustering"].le(params.clustering_threshold).to_numpy()
        drift_sign = np.sign(indicators["drift"].fillna(0).to_numpy())
        direction = np.where(follow, drift_sign, -drift_sign).astype(int)
        direction[~ready] = 0
        long_allowed = indicators["valid_close"].gt(indicators["ma200"]).to_numpy()
        short_allowed = indicators["valid_close"].lt(indicators["ma200"]).to_numpy()
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            if position != 0:
                # Only an opposite topology-conditioned direction closes a position.
                if direction[i] == -position:
                    position = 0
            elif ready[i]:
                if direction[i] == 1 and long_allowed[i]:
                    position = 1
                elif direction[i] == -1 and short_allowed[i]:
                    position = -1
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=300, freq="B")
    close = 100 + np.arange(300) * 0.04 + np.sin(np.arange(300) * 0.6)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:240], params)
    pd.testing.assert_frame_equal(indicators.iloc[:240], prefix)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    full = strategy.generate_signals(data, indicators, ctx, params).data
    partial = strategy.generate_signals(data.iloc[:240], prefix, ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:240], partial)
    assert full["signal"].iloc[:200].eq(0).all()
    assert full["signal"].isin([-1, 0, 1]).all()
    assert full["size"].gt(0).all()
    sample = data.iloc[:6]
    artificial = pd.DataFrame({
        "clustering": [0.0] * 6,
        "ma200": [90.0, 90.0, 90.0, 110.0, 110.0, 90.0],
        "drift": [1.0, 1.0, 0.0, -1.0, -1.0, 1.0],
        "valid_close": [100.0] * 6,
    }, index=sample.index)
    result = strategy.generate_signals(sample, artificial, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1]
    tiny = data.iloc[:3].copy()
    tiny["close"] = [3.0, 1.0, 3.0]
    assert strategy.indicators(tiny, StrategyParams(graph_window=3))["transitivity"].iloc[-1] == 1.0
    tiny["close"] = [1.0, 2.0, 3.0]
    assert strategy.indicators(tiny, StrategyParams(graph_window=3))["transitivity"].iloc[-1] == 0.0


if __name__ == "__main__":
    _self_check()
