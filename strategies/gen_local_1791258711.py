from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionReversalParams:
    min_gap_range: float = 0.10
    gap_dominance: float = 1.25
    min_retention: float = 0.70


class GeneratedStrategy(BaseStrategy[AuctionReversalParams]):
    strategy_id = "gen_local_1791258711"

    @classmethod
    def params_type(cls) -> type[AuctionReversalParams]:
        return AuctionReversalParams

    @staticmethod
    def warmup_bars(params: AuctionReversalParams) -> int:
        values = (params.min_gap_range, params.gap_dominance, params.min_retention)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("All parameters must be finite and positive")
        return 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: AuctionReversalParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_range"] = (data["high"] - data["low"]).shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["prior_gap"] = out["gap"].shift(1)
        range_denominator = out["prior_range"].where(out["prior_range"] > 0)
        gap_denominator = out["gap"].where(out["gap"] > 0)
        out["gap_range"] = out["gap"] / range_denominator
        out["retention"] = (data["close"] - out["prior_close"]) / gap_denominator
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionReversalParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & data[["open", "close"]].notna().all(axis=1)
        entry = (
            valid
            & (indicators["prior_gap"] < 0)
            & (indicators["gap"] > params.gap_dominance * indicators["prior_gap"].abs())
            & (indicators["gap_range"] >= params.min_gap_range)
            & (data["open"] > indicators["prior_high"])
            & (data["close"] > indicators["prior_high"])
            & (indicators["retention"] >= params.min_retention)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Exposure exits precisely when the same entry condition becomes false.
        df["signal"] = entry.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "open": [100.0, 99.0, 102.0, 103.0, 102.0],
            "high": [101.0, 101.0, 104.0, 104.0, 103.0],
            "low": [99.0, 98.0, 102.0, 102.0, 101.0],
            "close": [100.0, 100.0, 103.0, 103.0, 102.0],
            "volume": [1000.0] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    params = AuctionReversalParams()
    derived = GeneratedStrategy.indicators(sample, params)
    result = GeneratedStrategy.generate_signals(sample, derived, None, params)
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 0]
    assert result.data.index.equals(sample.index)
    assert (result.data["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
