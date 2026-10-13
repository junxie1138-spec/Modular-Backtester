from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 5
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1790933879"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return max(21, params.graph_window + 2)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.graph_window < 3 or not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("graph_window must be >= 3 and target_vol must be positive")
        w = params.graph_window
        pressure = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        nodes = [pressure.shift(k) for k in range(w)]
        valid = pressure.rolling(w, min_periods=w).count().eq(w)
        counts = []
        # Horizontal visibility: every interior node is below both endpoints.
        # ponytail: O(w^3) motif enumeration; keep windows small, use a stack for large windows.
        for polarity in (1, -1):
            heights = [polarity * node for node in nodes]
            edges = {}
            for a in range(w - 1):
                ceiling = heights[a + 1]
                for b in range(a + 1, w):
                    if b == a + 1:
                        edges[a, b] = heights[a].notna() & heights[b].notna()
                    else:
                        edges[a, b] = ceiling.lt(heights[a]) & ceiling.lt(heights[b])
                    ceiling = pd.concat([ceiling, heights[b]], axis=1).max(axis=1, skipna=False)
            triangles = pd.Series(0.0, index=data.index)
            for a in range(w - 2):
                for b in range(a + 1, w - 1):
                    for c in range(b + 1, w):
                        triangles += (edges[a, b] & edges[b, c] & edges[a, c]).astype(float)
            counts.append(triangles.where(valid))
        imbalance = (counts[0] - counts[1]) / max(1, (w - 1) * (w - 2) / 2)
        score = imbalance.rolling(3, min_periods=3).mean()
        returns = data["close"].where(data["close"] > 0).pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        annual_vol = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        ready = score.notna() & annual_vol.notna()
        fraction = (params.target_vol / annual_vol.clip(lower=0.01)).clip(0.05, 0.95)
        return pd.DataFrame({
            "pressure": pressure.fillna(0.0),
            "valley_triangles": counts[0].fillna(0.0),
            "peak_triangles": counts[1].fillna(0.0),
            "imbalance": imbalance.fillna(0.0),
            "score": score.fillna(0.0),
            "return": returns.fillna(0.0),
            "annual_vol": annual_vol.fillna(0.0),
            "ready": ready.astype(int),
            "fraction": fraction.fillna(0.05),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Remain long while valley cycles dominate; exit when that condition flips.
        entry = indicators["score"].gt(0) & indicators["ready"].eq(1)
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["fraction"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 240)))
    opening = close / (1 + rng.uniform(-0.015, 0.015, 240))
    bars = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * 1.002,
        "low": np.minimum(opening, close) * 0.998,
        "close": close,
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=240, freq="B"))
    strategy = GeneratedStrategy()
    params = TopologyParams()
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    features = strategy.indicators(bars, params)
    result = strategy.generate_signals(bars, features, context, params).data
    expected = (features["score"].gt(0) & features["ready"].eq(1)).astype(int)
    assert result["signal"].equals(expected.shift(1).fillna(0).astype(int))
    assert result["signal"].eq(1).any() and result["signal"].eq(0).any()
    assert result["size"].between(0.05, 0.95).all()
    prefix = bars.iloc[:150]
    prefix_features = strategy.indicators(prefix, params)
    prefix_result = strategy.generate_signals(prefix, prefix_features, context, params).data
    assert prefix_result.equals(result.iloc[:150])
