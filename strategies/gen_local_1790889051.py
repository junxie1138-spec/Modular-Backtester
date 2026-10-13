from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    lookback: int = 6
    acceleration_multiple: float = 1.2


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local_1790889051"

    @classmethod
    def params_type(cls):
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return params.lookback + 3

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not np.isfinite(params.acceleration_multiple) or params.acceleration_multiple <= 0:
            raise ValueError("acceleration_multiple must be positive and finite")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        out["roc"] = data["close"].div(previous_close).sub(1.0).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scale"] = out["acceleration"].abs().rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["ceiling"] = data["high"].shift(1).rolling(params.lookback, min_periods=params.lookback).max()
        out["opening_gap"] = data["open"].div(previous_close).sub(1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry = (
            ready
            & data["close"].gt(indicators["ceiling"])
            & indicators["roc"].gt(0.0)
            & indicators["acceleration"].gt(params.acceleration_multiple * indicators["acceleration_scale"])
            & indicators["opening_gap"].le(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The position exits exactly when the entry predicate becomes false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 103.0, 104.0, 104.0], index=pd.date_range("2020-01-01", periods=9))
    opening = close.shift(1).fillna(100.0) - 0.1
    data = pd.DataFrame({"open": opening, "high": close + 0.1, "low": opening - 0.1, "close": close, "volume": 1000000.0})
    strategy = GeneratedStrategy()
    params = BreakoutParams(lookback=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert signals["size"].gt(0.0).all()
