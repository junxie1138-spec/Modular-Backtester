from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionParams:
    percentile_window: int = 63
    expansion_quantile: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionParams]):
    strategy_id = "gen_local-w2_1791320130"

    @classmethod
    def params_type(cls) -> type[AuctionParams]:
        return AuctionParams

    @staticmethod
    def warmup_bars(params: AuctionParams) -> int:
        if params.percentile_window < 2:
            raise ValueError("percentile_window must be at least 2")
        if not 0.0 < params.expansion_quantile < 1.0:
            raise ValueError("expansion_quantile must be between 0 and 1")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        return int(params.percentile_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AuctionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["prior_range"] = out["range"].shift(1)
        out["expansion"] = (out["range"] / out["prior_range"]).replace([np.inf, -np.inf], np.nan)
        out["expansion_threshold"] = out["expansion"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.expansion_quantile)
        out["outside_day"] = (
            (data["high"] > out["prior_high"])
            & (data["low"] < out["prior_low"])
        )
        out["exceptional"] = (
            out["expansion"].notna()
            & out["expansion_threshold"].notna()
            & (out["expansion"] > out["expansion_threshold"])
        )
        out["entry"] = np.select(
            [
                out["outside_day"] & out["exceptional"] & (data["close"] > out["prior_high"]),
                out["outside_day"] & out["exceptional"] & (data["close"] < out["prior_low"]),
            ],
            [1, -1], default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, entry in enumerate(entries):
            if position and i == exit_at:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
                exit_at = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({"entry": [1, -1, 1, -1, 1, 1, 1, 0]}, index=index)
    for holding, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, -1, -1, 0, 1]),
    ):
        result = GeneratedStrategy.generate_signals(
            data, derived, None, AuctionParams(hold_bars=holding)
        )
        assert result.data["signal"].tolist() == expected
        assert (result.data["size"] > 0).all()
