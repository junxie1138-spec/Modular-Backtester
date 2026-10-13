from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    lookback: int = 6
    entry_ratio: float = 0.18
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1791248016"

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    @staticmethod
    def warmup_bars(params: MomentumParams) -> int:
        if not 3 <= params.lookback <= 9:
            raise ValueError("lookback must be between 3 and 9")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not 0.0 < params.entry_ratio < 1.0:
            raise ValueError("entry_ratio must be between zero and one")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = out["return"].rolling(params.lookback, min_periods=params.lookback)
        out["total_return"] = window.sum()
        out["gross_return"] = out["return"].abs().rolling(params.lookback, min_periods=params.lookback).sum()
        out["largest_gain"] = window.max().clip(lower=0.0)
        out["largest_loss"] = window.min().clip(upper=0.0)
        denominator = out["gross_return"].where(out["gross_return"] > 0.0)
        out["long_score"] = ((out["total_return"] - out["largest_gain"]) / denominator).fillna(0.0)
        out["short_score"] = ((out["largest_loss"] - out["total_return"]) / denominator).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        self.warmup_bars(params)
        long_entry = (indicators["long_score"] >= params.entry_ratio) & (indicators["return"] > 0.0)
        short_entry = (indicators["short_score"] >= params.entry_ratio) & (indicators["return"] < 0.0)
        candidates = np.where(long_entry.fillna(False), 1, np.where(short_entry.fillna(False), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for i, candidate in enumerate(candidates):
            if position != 0:
                remaining -= 1
                if remaining == 0:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                remaining = params.hold_bars
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = MomentumParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    for direction in (1, -1):
        features = pd.DataFrame({"return": float(direction), "long_score": float(direction == 1), "short_score": float(direction == -1)}, index=index)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [direction * x for x in [0, 1, 1, 1, 1, 0, 1, 1]]
        assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) <= 10
