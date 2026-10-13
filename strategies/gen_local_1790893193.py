from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compression_window: int = 4
    rank_window: int = 60
    compression_percentile: float = 0.35
    demand_percentile: float = 0.60


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790893193"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.compression_window + params.rank_window - 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        w, r = params.compression_window, params.rank_window
        ranges = (data["high"] - data["low"]).clip(lower=0)
        total_range = ranges.rolling(w, min_periods=w).sum().replace(0, np.nan)
        envelope = (
            data["high"].rolling(w, min_periods=w).max()
            - data["low"].rolling(w, min_periods=w).min()
        )
        recycling = envelope / total_range
        demand = (
            (data["close"] - data["open"]).rolling(w, min_periods=w).sum()
            / total_range
        )
        return pd.DataFrame(
            {
                "daily_range": ranges,
                "total_range": total_range,
                "envelope": envelope,
                "recycling": recycling,
                "demand": demand,
                "compression_rank": recycling.rolling(r, min_periods=r).rank(pct=True),
                "demand_rank": demand.rolling(r, min_periods=r).rank(pct=True),
            },
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        # Maintain exposure exactly while the entry condition remains true.
        condition = (
            indicators["compression_rank"].le(params.compression_percentile)
            & indicators["demand_rank"].ge(params.demand_percentile)
        )
        df = pd.DataFrame(
            {"signal": condition.astype(int), "size": 0.95}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "volume": 1000.0}, index=index
    )
    ranks = pd.DataFrame(
        {"compression_rank": [np.nan, 0.2, 0.2, 0.9, 0.9],
         "demand_rank": [np.nan, 0.8, 0.8, 0.8, 0.8]}, index=index
    )
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(bars, ranks, context, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
