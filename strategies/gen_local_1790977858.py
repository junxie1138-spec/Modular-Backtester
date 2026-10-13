from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeOrderParams:
    shape_bars: int = 10
    history_bars: int = 504
    forecast_bars: int = 18
    neighbors: int = 12
    reversal_threshold: float = 0.003


class GeneratedStrategy(BaseStrategy[ShapeOrderParams]):
    strategy_id = "gen_local_1790977858"

    @classmethod
    def params_type(cls):
        return ShapeOrderParams

    def warmup_bars(self, params):
        if params.shape_bars < 2 or params.forecast_bars < 1:
            raise ValueError("Shape needs at least two bars; forecast must be positive.")
        if not 1 <= params.neighbors <= params.history_bars:
            raise ValueError("Neighbors must be between one and history_bars.")
        if not np.isfinite(params.reversal_threshold) or params.reversal_threshold <= 0:
            raise ValueError("Reversal threshold must be finite and positive.")
        return int(params.history_bars + params.forecast_bars + 2 * params.shape_bars + 1)

    def indicators(self, data, params):
        warmup = self.warmup_bars(params)
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        width = params.shape_bars
        out = pd.DataFrame(index=data.index)
        columns = []
        for lag in range(width - 1, -1, -1):
            name = f"return_lag_{lag}"
            out[name] = returns.shift(lag)
            columns.append(name)
        out["realized_horizon_return"] = (close / close.shift(params.forecast_bars) - 1).replace([np.inf, -np.inf], np.nan)
        paths = out[columns].to_numpy(dtype=float)
        realized = out["realized_horizon_return"].to_numpy(dtype=float)
        premium = np.full(len(data), np.nan)
        # ponytail: O(bars * history * shape); chunk distances if histories grow substantially.
        for t in range(warmup - 1, len(data)):
            query = paths[t]
            if not np.isfinite(query).all():
                continue
            # All analog outcomes end before the current shape begins.
            last = t - params.forecast_bars - width
            ends = np.arange(last - params.history_bars + 1, last + 1)
            candidates = paths[ends]
            outcomes = realized[ends + params.forecast_bars]
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(outcomes)
            candidates, outcomes = candidates[valid], outcomes[valid]
            k = params.neighbors
            if len(outcomes) < k:
                continue
            chronological = np.mean((candidates - query) ** 2, axis=1)
            reversed_order = np.mean((candidates - query[::-1]) ** 2, axis=1)
            first = np.argsort(chronological, kind="stable")[:k]
            second = np.argsort(reversed_order, kind="stable")[:k]
            premium[t] = 0.5 * (outcomes[first].mean() - outcomes[second].mean())
        out["order_premium"] = premium
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        threshold = params.reversal_threshold
        score = indicators["order_premium"]
        # Equal and opposite evidence changes state; missing evidence preserves it.
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[score.ge(threshold)] = 1.0
        state.loc[score.le(-threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ShapeOrderParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=640, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.01, len(index))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:610], params)
    pd.testing.assert_frame_equal(full.iloc[:610], prefix)
    assert full["order_premium"].notna().any()
    evidence = pd.DataFrame({"order_premium": [0.0, 0.02, np.nan, -0.02, 0.02]}, index=index[:5])
    signals = strategy.generate_signals(data.iloc[:5], evidence, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0]
    assert signals["size"].gt(0).all()
