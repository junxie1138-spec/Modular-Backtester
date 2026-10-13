from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeGapParams:
    lookback: int = 8
    min_gap_fraction: float = 0.05


class GeneratedStrategy(BaseStrategy[RangeGapParams]):
    strategy_id = "gen_local_1791208152"

    @classmethod
    def params_type(cls) -> type[RangeGapParams]:
        return RangeGapParams

    @staticmethod
    def warmup_bars(params: RangeGapParams) -> int:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not np.isfinite(params.min_gap_fraction) or params.min_gap_fraction <= 0:
            raise ValueError("min_gap_fraction must be finite and positive")
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeGapParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        df["prior_high"] = data["high"].rolling(params.lookback, min_periods=params.lookback).max().shift(1)
        df["prior_low"] = data["low"].rolling(params.lookback, min_periods=params.lookback).min().shift(1)
        df["prior_span"] = (df["prior_high"] - df["prior_low"]).where(df["prior_high"] > df["prior_low"])
        df["midpoint"] = (df["prior_high"] + df["prior_low"]) / 2.0
        df["prior_close"] = data["close"].shift(1)
        df["gap_fraction"] = (data["open"] - df["prior_close"]) / df["prior_span"]
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeGapParams) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & np.isfinite(data["open"]) & np.isfinite(data["close"])
        entry_condition = (
            valid
            & (indicators["prior_close"] <= indicators["midpoint"])
            & (data["open"] > indicators["midpoint"])
            & (data["close"] > indicators["midpoint"])
            & (data["close"] < indicators["prior_high"])
            & (indicators["gap_fraction"] >= params.min_gap_fraction)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Target exposure becomes zero only when the entry condition is false.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "open": [99.0, 99.0, 101.0, 101.0, 101.0],
            "high": [102.0] * 5,
            "low": [98.0] * 5,
            "close": [99.0, 99.0, 101.0, 101.0, 101.0],
            "volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-01-01", periods=5),
    )
    params = RangeGapParams(lookback=2)
    derived = GeneratedStrategy.indicators(sample, params)
    result = GeneratedStrategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 3
