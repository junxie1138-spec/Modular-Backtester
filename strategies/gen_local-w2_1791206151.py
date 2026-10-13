from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphMedoidParams:
    window: int = 9
    displacement: float = 0.15


class GeneratedStrategy(BaseStrategy[GraphMedoidParams]):
    strategy_id = "gen_local-w2_1791206151"

    @classmethod
    def params_type(cls) -> type[GraphMedoidParams]:
        return GraphMedoidParams

    def warmup_bars(self, params: GraphMedoidParams) -> int:
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: GraphMedoidParams) -> pd.DataFrame:
        if isinstance(params.window, bool) or not isinstance(params.window, (int, np.integer)) or params.window < 3:
            raise ValueError("window must be an integer of at least 3")
        if not np.isfinite(params.displacement) or not 0.0 < params.displacement < 1.0:
            raise ValueError("displacement must be finite and between zero and one")
        w = int(params.window)
        close = data["close"].astype(float)
        prices = np.column_stack([close.shift(w - 1 - i).to_numpy() for i in range(w)])
        valid = np.isfinite(prices).all(axis=1)
        distances = np.full((len(data), w, w), np.inf)
        diagonal = np.arange(w)
        distances[:, diagonal, diagonal] = 0.0
        # Natural visibility: every intermediate close lies below the connecting chord.
        for i in range(w - 1):
            for j in range(i + 1, w):
                visible = valid.copy()
                if j > i + 1:
                    fractions = np.arange(1, j - i, dtype=float) / (j - i)
                    chord = prices[:, i, None] + (prices[:, j] - prices[:, i])[:, None] * fractions
                    visible &= np.all(prices[:, i + 1:j] < chord, axis=1)
                edge = np.where(visible, 1.0, np.inf)
                distances[:, i, j] = edge
                distances[:, j, i] = edge
        # ponytail: O(bars * window^3); use sparse shortest paths if windows become large.
        for k in range(w):
            distances = np.minimum(distances, distances[:, :, k, None] + distances[:, None, k, :])
        costs = distances.sum(axis=2)
        minimum = costs.min(axis=1)
        medoids = (costs == minimum[:, None]) & valid[:, None]
        counts = medoids.sum(axis=1)
        anchor = np.sum(np.where(medoids, prices, 0.0), axis=1) / np.maximum(counts, 1)
        anchor = np.where(valid, anchor, np.nan)
        span = close.rolling(w, min_periods=w).max() - close.rolling(w, min_periods=w).min()
        result = pd.DataFrame(index=data.index)
        result["graph_anchor"] = anchor
        result["close_span"] = span
        result["relative_position"] = ((close - result["graph_anchor"]) / span.where(span > 0.0)).replace([np.inf, -np.inf], np.nan)
        result["medoid_count"] = np.where(valid, counts, np.nan)
        result["medoid_mean_distance"] = np.where(valid, minimum / (w - 1), np.nan)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GraphMedoidParams) -> SignalFrame:
        position = indicators["relative_position"]
        candidates = np.select(
            [position.ge(params.displacement), position.le(-params.displacement)],
            [1, -1],
            default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        state = 0
        for i, candidate in enumerate(candidates):
            if state != 0 and candidate == -state:
                state = 0
            elif state == 0 and candidate != 0:
                state = int(candidate)
            signals[i] = state
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GraphMedoidParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    readings = pd.DataFrame({"relative_position": [np.nan, 0.3, 0.0, -0.3, -0.3, 0.0, 0.3]}, index=index)
    output = strategy.generate_signals(data, readings, None, params).data
    assert output["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert output["size"].gt(0.0).all()
    assert strategy.indicators(data, params)["relative_position"].isna().all()
