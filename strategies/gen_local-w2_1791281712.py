from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 6
    min_drawdown: float = 0.003
    curvature_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791281712"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if not isinstance(params.lookback, int) or not 4 <= params.lookback <= 10:
            raise ValueError("lookback must be an integer between 4 and 10")
        if not np.isfinite(params.min_drawdown) or not 0 < params.min_drawdown < 1:
            raise ValueError("min_drawdown must be between zero and one")
        if not np.isfinite(params.curvature_threshold) or params.curvature_threshold <= 0:
            raise ValueError("curvature_threshold must be positive")
        return max(params.lookback, 4)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = data["high"].rolling(params.lookback, min_periods=params.lookback).max()
        out["trough"] = data["low"].rolling(params.lookback, min_periods=params.lookback).min()
        out["roc"] = close.pct_change(2, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["drawdown"] = (out["peak"] - close) / out["peak"].where(out["peak"] > 0)
        span = (out["peak"] - out["trough"]).where(out["peak"] > out["trough"])
        out["recovery"] = (close - out["trough"]) / span
        out["curvature"] = out["acceleration"] / out["drawdown"].where(out["drawdown"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        eligible = (
            indicators["drawdown"].ge(params.min_drawdown)
            & indicators["recovery"].between(0.2, 0.8)
        )
        bullish = (eligible & indicators["curvature"].ge(params.curvature_threshold)).fillna(False)
        bearish = (eligible & indicators["curvature"].le(-params.curvature_threshold)).fillna(False)
        candidates = np.select([bullish, bearish], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 99, 98, 97, 96, 97, 98, 97, 96], dtype=float)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert actual["drawdown"].iloc[:5].isna().all()
    signals = strategy.generate_signals(data, actual, ctx, params).data
    assert signals["signal"].iloc[:6].eq(0).all()
    evidence = pd.DataFrame(
        {"drawdown": 0.01, "recovery": 0.5,
         "curvature": [0.2, 0.0, 0.2, -0.2, -0.2, 0.0, 0.2, 0.2, 0.0]},
        index=index,
    )
    result = strategy.generate_signals(data, evidence, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
    for lookback in (5, 7, 9):
        assert strategy.warmup_bars(RecoveryParams(lookback=lookback)) <= 10


if __name__ == "__main__":
    _self_check()
