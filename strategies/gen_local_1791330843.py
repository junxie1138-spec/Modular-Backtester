from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphParams:
    graph_window: int = 18
    min_clustering: float = 0.35
    atr_window: int = 14
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GraphParams]):
    strategy_id = "gen_local_1791330843"

    @classmethod
    def params_type(cls):
        return GraphParams

    def warmup_bars(self, params: GraphParams) -> int:
        if params.graph_window < 3 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; graph_window >= 3.")
        if not 0.0 <= params.min_clustering < 1.0:
            raise ValueError("min_clustering must be in [0, 1).")
        return int(max(params.graph_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
        self.warmup_bars(params)
        w = params.graph_window
        close = data["close"].astype(float)
        prices = pd.concat([close.shift(w - j) for j in range(w)], axis=1).to_numpy()
        valid = np.isfinite(prices).all(axis=1)
        adjacency = np.zeros((len(data), w, w), dtype=float)
        for left in range(w - 1):
            adjacency[:, left, left + 1] = 1.0
            adjacency[:, left + 1, left] = 1.0
            for right in range(left + 2, w):
                visible = prices[:, left + 1:right].max(axis=1) < np.minimum(
                    prices[:, left], prices[:, right]
                )
                adjacency[:, left, right] = visible
                adjacency[:, right, left] = visible
        degree = adjacency.sum(axis=2)
        wedges = (degree * (degree - 1.0) / 2.0).sum(axis=1)
        # ponytail: dense O(bars * window**3) triangle counting; use sparse graphs for large windows.
        closed_wedges = ((adjacency @ adjacency) * adjacency).sum(axis=(1, 2)) / 2.0
        clustering = np.divide(closed_wedges, wedges, out=np.zeros(len(data)), where=wedges > 0)
        clustering[~valid] = np.nan
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["clustering"] = clustering
        out["upper"] = data["high"].rolling(w, min_periods=w).max().shift(1)
        out["lower"] = data["low"].rolling(w, min_periods=w).min().shift(1)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean().shift(1)
        ready = np.isfinite(out).all(axis=1) & np.isfinite(close) & out["atr"].gt(0)
        gate = ready & out["clustering"].ge(params.min_clustering)
        out["candidate"] = np.select([
            gate & close.gt(out["upper"]),
            gate & close.lt(out["lower"]),
        ], [1, -1], default=0).astype(int)
        excess = pd.concat([close - out["upper"], out["lower"] - close], axis=1).max(axis=1)
        out["breakout_strength"] = (excess / out["atr"].replace(0, np.nan)).clip(0, 1).fillna(0)
        out["topology_strength"] = (
            (out["clustering"] - params.min_clustering) / (1.0 - params.min_clustering)
        ).clip(0, 1).fillna(0)
        out["entry_size"] = (
            0.25 + 0.375 * (out["breakout_strength"] + out["topology_strength"])
        ).clip(0.25, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GraphParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.25).clip(0.25, 1.0).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        side, exit_bar, held_size = 0, -1, 0.25
        for i in range(len(data)):
            if side != 0 and i == exit_bar:
                side = 0
                continue
            if side == 0 and candidates[i] != 0:
                side = int(candidates[i])
                held_size = float(entry_sizes[i])
                exit_bar = i + params.hold_bars
            if side != 0:
                signal[i] = side
                size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    p = GraphParams(graph_window=5, atr_window=3, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=40)
    close = pd.Series(np.arange(40, dtype=float) + 100.0, index=index)
    data = pd.DataFrame({"open": close, "high": close + 0.2,
                         "low": close - 0.2, "close": close, "volume": 1000.0})
    ind = strategy.indicators(data, p)
    assert ind["clustering"].iloc[5:].eq(0).all()
    pd.testing.assert_frame_equal(ind.iloc[:25], strategy.indicators(data.iloc[:25], p))
    oscillating = data.copy()
    oscillating["close"] = 100.0 + np.tile([0.0, 2.0], 20)
    assert strategy.indicators(oscillating, p)["clustering"].max() > 0.0
    decisions = pd.DataFrame({
        "candidate": [1, -1, -1, -1, -1, 1, 1, 1, 0, 0],
        "entry_size": [0.4] * 4 + [0.8] * 6,
    }, index=index[:10])
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    result = strategy.generate_signals(data.iloc[:10], decisions, ctx, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 0]
    assert np.allclose(result["size"].iloc[1:4], 0.4)
    assert np.allclose(result["size"].iloc[5:8], 0.8)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
