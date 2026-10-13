from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverflowParams:
    roc_bars: int = 2
    capacity_bars: int = 16
    entry_strength: float = 0.5


class GeneratedStrategy(BaseStrategy[OverflowParams]):
    strategy_id = "gen_local_1790904607"

    @classmethod
    def params_type(cls):
        return OverflowParams

    def warmup_bars(self, params: OverflowParams) -> int:
        return int(params.roc_bars + params.capacity_bars + 2)

    def indicators(self, data: pd.DataFrame, params: OverflowParams) -> pd.DataFrame:
        if params.roc_bars < 1 or params.capacity_bars < 2 or not np.isfinite(params.entry_strength) or params.entry_strength <= 0:
            raise ValueError("Lookbacks must be positive and entry_strength must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["roc"] = close.div(close.shift(params.roc_bars)).sub(1)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scale"] = out["acceleration"].abs().rolling(params.capacity_bars, min_periods=params.capacity_bars).mean().shift(1)
        out["volume_capacity"] = volume.rolling(params.capacity_bars, min_periods=params.capacity_bars).mean().shift(1)
        out["overflow"] = volume.div(out["volume_capacity"].where(out["volume_capacity"] > 0)).sub(1).clip(lower=0)
        out["strength"] = out["acceleration"].div(out["acceleration_scale"].where(out["acceleration_scale"] > 0)).mul(out["overflow"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OverflowParams) -> SignalFrame:
        strength = indicators["strength"].fillna(0)
        entry = indicators["roc"].gt(0) & strength.gt(params.entry_strength)
        df = pd.DataFrame(index=data.index)
        # Reversal exit: remain long only while the identical entry condition holds.
        df["signal"] = entry.astype(int)
        conviction = strength.div(params.entry_strength).sub(1).clip(lower=0)
        df["size"] = 0.2 + 0.75 * (1.0 - np.exp(-conviction))
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = OverflowParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=100)
    t = np.arange(100)
    close = 100 * np.exp(np.cumsum(0.002 + 0.004 * np.sin(t)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": np.where(t % 3 == 0, 3000000.0, 1000000.0)}, index=index)
    full = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    prefix = data.iloc[:70]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(full.iloc[:70], partial)
    assert full["signal"].isin([0, 1]).all() and full["signal"].eq(1).any()
    assert full["size"].between(0.2, 0.95).all()
