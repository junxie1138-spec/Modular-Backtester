from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownAuctionParams:
    lookback: int = 5
    minimum_creation: float = 0.25
    repair_band: float = 0.15


class GeneratedStrategy(BaseStrategy[DrawdownAuctionParams]):
    strategy_id = "gen_local-w2_1791307632"

    @classmethod
    def params_type(cls) -> type[DrawdownAuctionParams]:
        return DrawdownAuctionParams

    @staticmethod
    def warmup_bars(params: DrawdownAuctionParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not np.isfinite(params.minimum_creation) or params.minimum_creation <= 0:
            raise ValueError("minimum_creation must be positive and finite")
        if not np.isfinite(params.repair_band) or not 0 < params.repair_band < 0.5:
            raise ValueError("repair_band must be between 0 and 0.5")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownAuctionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["peak"] = data["high"].rolling(params.lookback, min_periods=params.lookback).max().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["range_scale"] = out["true_range"].rolling(params.lookback, min_periods=params.lookback).mean().shift(1).replace(0.0, np.nan)
        out["previous_depth"] = (out["peak"] - out["previous_close"]).clip(lower=0.0)
        out["opening_depth"] = (out["peak"] - data["open"]).clip(lower=0.0)
        out["closing_depth"] = (out["peak"] - data["close"]).clip(lower=0.0)
        out["created_depth"] = out["opening_depth"] - out["previous_depth"]
        out["creation_scale"] = out["created_depth"] / out["range_scale"]
        out["repair_fraction"] = (out["opening_depth"] - out["closing_depth"]) / out["created_depth"].where(out["created_depth"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownAuctionParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        eligible = indicators["creation_scale"].ge(params.minimum_creation)
        long_condition = (eligible & indicators["repair_fraction"].gt(0.5 + params.repair_band)).fillna(False)
        short_condition = (eligible & indicators["repair_fraction"].lt(0.5 - params.repair_band)).fillna(False)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (go_long, go_short) in enumerate(zip(long_condition.to_numpy(), short_condition.to_numpy())):
            if position == 1:
                if not go_long:
                    position = 0
            elif position == -1:
                if not go_short:
                    position = 0
            elif go_long:
                position = 1
            elif go_short:
                position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = DrawdownAuctionParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"creation_scale": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "repair_fraction": [np.nan, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(DrawdownAuctionParams(lookback=9)) == 10
    calculated = GeneratedStrategy.indicators(None, data, params)
    assert calculated.index.equals(data.index)
    assert calculated["repair_fraction"].isna().all()


if __name__ == "__main__":
    _self_check()
