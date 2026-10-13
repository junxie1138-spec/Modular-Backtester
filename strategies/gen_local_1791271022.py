from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 10
    entry_atr: float = 0.85
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791271022"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.window < 2 or params.entry_atr <= 0:
            raise ValueError("window must be >= 2 and entry_atr must be positive")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.window).mean()
        out["mass"] = out["true_range"].pow(2)
        out["weighted_price"] = out["mass"] * out["midpoint"]
        out["mass_sum"] = out["mass"].rolling(params.window).sum()
        out["price_sum"] = out["weighted_price"].rolling(params.window).sum()
        out["centroid"] = out["price_sum"] / out["mass_sum"].replace(0, np.nan)
        out["score"] = ((out["centroid"] - data["close"]) /
                        out["atr"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        scores = indicators["score"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_size = params.min_size
        for i, score in enumerate(scores):
            setup = 0
            if np.isfinite(score):
                if score >= params.entry_atr:
                    setup = 1
                elif score <= -params.entry_atr:
                    setup = -1
            if position and setup == -position:
                position = 0
                entry_size = params.min_size
            elif position == 0 and setup:
                position = setup
                conviction = min(abs(score) / (2.0 * params.entry_atr), 1.0)
                entry_size = params.min_size + (params.max_size - params.min_size) * conviction
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    close = 100.0 + 3.0 * np.sin(np.arange(80) / 3.0)
    data = pd.DataFrame({
        "open": close, "high": close + 0.6, "low": close - 0.6,
        "close": close, "volume": 1000000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, ctx, params).data
    prefix = data.iloc[:47]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:47], partial)
    fixture = pd.DataFrame({"score": [np.nan, 1.0, 0.0, -2.0, -2.0, 0.0, 1.0, 1.0]}, index=index[:8])
    checked = strategy.generate_signals(data.iloc[:8], fixture, ctx, params).data
    assert checked["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert checked["size"].iloc[5] > checked["size"].iloc[2] > params.min_size
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    assert result["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
