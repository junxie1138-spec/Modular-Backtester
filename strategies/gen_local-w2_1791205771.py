from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    lookback: int = 7
    motif_threshold: float = 0.2
    min_move: float = 0.003


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791205771"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(params.lookback)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        w = params.lookback
        if not 4 <= w <= 16:
            raise ValueError("lookback must be between 4 and 16")
        if not 0 < params.motif_threshold <= 1 or not 0 <= params.min_move < 1:
            raise ValueError("Invalid motif_threshold or min_move")
        out = pd.DataFrame(np.nan, index=data.index,
                           columns=["upper_triangles", "lower_triangles", "motif_balance", "move"])
        if len(data) < w:
            return out
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0)).to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(log_close, w)
        valid = np.isfinite(windows).all(axis=1)
        n = len(windows)
        upper = np.zeros((n, w, w), dtype=float)
        lower = np.zeros_like(upper)
        # Natural visibility: each intermediate point lies strictly below the edge.
        # Reflecting log prices supplies the symmetric lower visibility graph.
        # ponytail: dense O(n*w^3) motifs; sparse counting if windows exceed 16.
        for i in range(w - 1):
            for j in range(i + 1, w):
                if j == i + 1:
                    up = down = np.ones(n, dtype=bool)
                else:
                    fraction = np.arange(1, j - i) / (j - i)
                    line = windows[:, i, None] + (windows[:, j] - windows[:, i])[:, None] * fraction
                    middle = windows[:, i + 1:j]
                    up = (middle < line).all(axis=1)
                    down = (middle > line).all(axis=1)
                upper[:, i, j] = upper[:, j, i] = up
                lower[:, i, j] = lower[:, j, i] = down
        possible = w * (w - 1) * (w - 2) / 6
        up_density = np.einsum("nij,njk,nki->n", upper, upper, upper, optimize=True) / (6 * possible)
        down_density = np.einsum("nij,njk,nki->n", lower, lower, lower, optimize=True) / (6 * possible)
        values = np.column_stack((up_density, down_density, up_density - down_density,
                                  windows[:, -1] - windows[:, 0]))
        values[~valid] = np.nan
        out.iloc[w - 1:] = values
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        balance = indicators["motif_balance"]
        move = indicators["move"]
        long_entry = (balance >= params.motif_threshold) & (move <= -params.min_move)
        short_entry = (balance <= -params.motif_threshold) & (move >= params.min_move)
        intent = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, desired in enumerate(intent):
            if position and desired == -position:
                position = 0
            elif position == 0 and desired:
                position = int(desired)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": 100.0}, index=index)
    features = pd.DataFrame({"motif_balance": [0.5, 0, -0.5, -0.5, 0.5, 0.5],
                             "move": [-0.02, 0, 0.02, 0.02, -0.02, -0.02]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    reflected = strategy.generate_signals(data, -features, ctx, params).data
    assert reflected["signal"].equals(-result["signal"])
    prices = pd.DataFrame({"close": np.exp(np.arange(12, dtype=float) ** 2 / 100)}, index=pd.date_range("2020-01-01", periods=12))
    motifs = strategy.indicators(prices, params)
    assert np.allclose(motifs["upper_triangles"].dropna(), 1.0)
    assert np.allclose(motifs["lower_triangles"].dropna(), 0.0)


if __name__ == "__main__":
    _self_check()
