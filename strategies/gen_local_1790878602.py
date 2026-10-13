from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnvelopeParams:
    window: int = 16
    history_bars: int = 504
    neighbors: int = 9
    holding_bars: int = 18
    minimum_return: float = 0.006


class GeneratedStrategy(BaseStrategy[EnvelopeParams]):
    strategy_id = "gen_local_1790878602"

    @classmethod
    def params_type(cls):
        return EnvelopeParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.window + params.holding_bars + params.neighbors))

    @staticmethod
    def indicators(data, params):
        if params.window < 6 or params.history_bars < params.neighbors:
            raise ValueError("Require window >= 6 and history_bars >= neighbors")
        if params.neighbors < 1 or params.holding_bars < 1:
            raise ValueError("Neighbors and holding_bars must be positive")
        w = int(params.window)
        n = len(data)
        names = ["edge_imbalance", "edge_disagreement", "short_overlap", "long_overlap", "triangle_imbalance", "hub_displacement"]
        out = pd.DataFrame(np.nan, index=data.index, columns=names + ["analog_return"])
        out["regime_ma"] = data["close"].rolling(200, min_periods=200).mean()
        if n < w:
            return out

        graphs = []
        valid = np.ones(n - w + 1, dtype=bool)
        for column in ("high", "low"):
            prices = data[column].to_numpy(dtype=float)
            positive = np.isfinite(prices) & (prices > 0)
            logged = np.log(np.where(positive, prices, np.nan))
            windows = np.lib.stride_tricks.sliding_window_view(logged, w)
            valid &= np.isfinite(windows).all(axis=1)
            graph = np.zeros((len(windows), w, w), dtype=float)
            for left in range(w - 1):
                for right in range(left + 1, w):
                    if right == left + 1:
                        visible = np.ones(len(windows), dtype=bool)
                    else:
                        fraction = np.arange(1, right - left, dtype=float) / (right - left)
                        line = windows[:, left, None] + (windows[:, right] - windows[:, left])[:, None] * fraction
                        visible = np.all(windows[:, left + 1:right] < line, axis=1)
                    graph[:, left, right] = visible
                    graph[:, right, left] = visible
            graphs.append(graph)

        upper, lower = graphs
        row, col = np.triu_indices(w, k=2)
        distances = col - row
        up_edges = upper[:, row, col]
        low_edges = lower[:, row, col]
        union = np.maximum(up_edges, low_edges)
        intersection = up_edges * low_edges
        short = distances <= 4
        long = distances > 4
        upper_triangles = ((upper @ upper) * upper).sum(axis=(1, 2)) / 6.0
        lower_triangles = ((lower @ lower) * lower).sum(axis=(1, 2)) / 6.0
        possible_triangles = w * (w - 1) * (w - 2) / 6.0
        features = np.column_stack([
            (up_edges - low_edges).mean(axis=1),
            np.abs(up_edges - low_edges).sum(axis=1) / np.maximum(union.sum(axis=1), 1.0),
            intersection[:, short].sum(axis=1) / np.maximum(union[:, short].sum(axis=1), 1.0),
            intersection[:, long].sum(axis=1) / np.maximum(union[:, long].sum(axis=1), 1.0),
            (upper_triangles - lower_triangles) / possible_triangles,
            (upper.sum(axis=2).argmax(axis=1) - lower.sum(axis=2).argmax(axis=1)) / float(w - 1),
        ])
        features[~valid] = np.nan
        out.loc[:, names] = np.vstack([np.full((w - 1, len(names)), np.nan), features])
        fingerprints = out[names].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        forecasts = np.full(n, np.nan)
        hold = int(params.holding_bars)
        k = int(params.neighbors)
        for t in range(w - 1, n):
            if not np.isfinite(fingerprints[t]).all():
                continue
            first = max(w - 1, t - int(params.history_bars))
            # An analog's next-open entry and timed-exit open must both be known by t.
            candidates = np.arange(first, t - hold, dtype=int)
            if len(candidates) < k:
                continue
            entry_opens = opens[candidates + 1]
            exit_opens = opens[candidates + hold + 1]
            usable = np.isfinite(fingerprints[candidates]).all(axis=1)
            usable &= np.isfinite(entry_opens) & np.isfinite(exit_opens)
            usable &= (entry_opens > 0) & (exit_opens > 0)
            candidates = candidates[usable]
            if len(candidates) < k:
                continue
            squared_distance = ((fingerprints[candidates] - fingerprints[t]) ** 2).mean(axis=1)
            nearest = np.argpartition(squared_distance, k - 1)[:k]
            selected = candidates[nearest]
            returns = opens[selected + hold + 1] / opens[selected + 1] - 1.0
            forecasts[t] = float(np.median(returns))
        out["analog_return"] = forecasts
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        eligible = (
            data["close"].gt(indicators["regime_ma"])
            & indicators["analog_return"].gt(params.minimum_return)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for t in range(len(data)):
            if remaining > 0:
                remaining -= 1
                signal[t] = int(remaining > 0)
            elif eligible[t]:
                remaining = int(params.holding_bars)
                signal[t] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = EnvelopeParams()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"regime_ma": 90.0, "analog_return": 0.02}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20:38].eq(1).all()
    assert result["signal"].iloc[38] == 0
    assert result["size"].gt(0).all()
    indicators["regime_ma"] = 110.0
    assert GeneratedStrategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
