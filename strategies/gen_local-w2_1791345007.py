from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBridgeParams:
    lookback: int = 20
    gap_multiple: float = 1.0


class GeneratedStrategy(BaseStrategy[GapBridgeParams]):
    strategy_id = "gen_local-w2_1791345007"

    @classmethod
    def params_type(cls) -> type[GapBridgeParams]:
        return GapBridgeParams

    @staticmethod
    def warmup_bars(params: GapBridgeParams) -> int:
        if params.lookback < 2 or not np.isfinite(params.gap_multiple) or params.gap_multiple <= 0:
            raise ValueError("lookback must be >= 2 and gap_multiple finite and positive")
        return int(params.lookback + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapBridgeParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_body_midpoint"] = (data["open"].shift(1) + out["prior_close"]) / 2.0
        out["gap"] = (data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].abs().rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["prior_bearish"] = out["prior_close"] < data["open"].shift(1)
        out["bridge_accepted"] = (data["open"] > out["prior_body_midpoint"]) & (data["low"] > out["prior_body_midpoint"])
        out["bullish_session"] = data["close"] > data["open"]
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBridgeParams,
    ) -> SignalFrame:
        entry = (
            indicators["gap_scale"].notna()
            & (indicators["gap_scale"] > 0)
            & (indicators["gap"] > params.gap_multiple * indicators["gap_scale"])
            & indicators["prior_bearish"]
            & indicators["bridge_accepted"]
            & indicators["bullish_session"]
        )
        df = pd.DataFrame(index=data.index)
        # The position exits only when this same entry predicate becomes false.
        # A bullish entry candle cannot also be the bearish predecessor required by the next entry.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapBridgeParams(lookback=3, gap_multiple=1.0)
    data = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 101, 102, 100],
            "high": [101, 101, 101, 101, 103, 103, 101],
            "low": [98, 98, 98, 98, 100, 100, 99],
            "close": [99, 99, 99, 99, 102, 101, 100],
            "volume": [1000] * 7,
        },
        index=pd.date_range("2020-01-01", periods=7),
    )
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert result.index.equals(data.index) and (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) == 5
    for end in range(1, len(data) + 1):
        prefix = data.iloc[:end]
        partial = GeneratedStrategy.indicators(prefix, params)
        assert partial.equals(derived.iloc[:end])


if __name__ == "__main__":
    _self_check()
