from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 63
    pullback_percentile: float = 0.35
    recovery_floor: float = 0.65


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790868515"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0.0 <= params.pullback_percentile <= 1.0:
            raise ValueError("pullback_percentile must lie in [0, 1]")
        if not 0.0 <= params.recovery_floor <= 1.0:
            raise ValueError("recovery_floor must lie in [0, 1]")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["return"] = (
            data["close"] / previous_close.where(previous_close > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["return"].rolling(
            params.lookback, min_periods=params.lookback
        ).rank(pct=True)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (
            (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["early_week"] = data.index.dayofweek <= 1
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        seasonal_pullback = (
            indicators["early_week"]
            & indicators["return_rank"].le(params.pullback_percentile)
        ).fillna(False)
        intraday_recovery = indicators["close_location"].ge(
            params.recovery_floor
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Exit exactly when the same two-part entry condition becomes false.
        df["signal"] = (seasonal_pullback & intraday_recovery).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-04", periods=7)
    close = np.arange(100.0, 93.0, -1.0)
    data = pd.DataFrame(
        {
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    strategy = GeneratedStrategy()
    params = Params(lookback=2, pullback_percentile=0.5, recovery_floor=0.6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=3)
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    derived.loc[index[3], "close_location"] = 0.2
    reversed_result = strategy.generate_signals(data, derived, ctx, params).data
    assert reversed_result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 3


if __name__ == "__main__":
    _self_check()
