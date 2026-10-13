from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionReleaseParams:
    lookback: int = 60
    volume_percentile: float = 0.75
    balance_percentile: float = 0.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[AbsorptionReleaseParams]):
    strategy_id = "gen_local-w2_1790983953"

    @classmethod
    def params_type(cls) -> type[AbsorptionReleaseParams]:
        return AbsorptionReleaseParams

    def warmup_bars(self, params: AbsorptionReleaseParams) -> int:
        return int(params.lookback) + 2

    def indicators(self, data: pd.DataFrame, params: AbsorptionReleaseParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0 < params.volume_percentile < 1 or not 0 < params.balance_percentile < 1:
            raise ValueError("Invalid percentile parameters")
        prices = data[["open", "high", "low", "close", "volume"]].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["valid"] = prices.notna().all(axis=1) & prices.gt(0).all(axis=1) & prices["high"].gt(prices["low"])
        out["range"] = (prices["high"] - prices["low"]).where(out["valid"])
        out["body_fraction"] = (prices["close"] - prices["open"]).abs() / out["range"]
        out["volume"] = prices["volume"].where(out["valid"])
        out["volume_cutoff"] = out["volume"].rolling(params.lookback, min_periods=params.lookback).quantile(params.volume_percentile).shift(1)
        out["balance_cutoff"] = out["body_fraction"].rolling(params.lookback, min_periods=params.lookback).quantile(params.balance_percentile).shift(1)
        out["absorption"] = out["valid"] & out["volume"].ge(out["volume_cutoff"]) & out["body_fraction"].le(out["balance_cutoff"])
        out["prior_absorption"] = out["absorption"].shift(1, fill_value=False)
        out["prior_high"] = prices["high"].shift(1)
        out["prior_low"] = prices["low"].shift(1)
        out["confirmed"] = out["valid"] & out["prior_absorption"] & out["volume"].ge(out["volume_cutoff"])
        out["long_entry"] = out["confirmed"] & prices["close"].gt(out["prior_high"]) & prices["close"].gt(prices["open"])
        out["short_entry"] = out["confirmed"] & prices["close"].lt(out["prior_low"]) & prices["close"].lt(prices["open"])
        out["entry"] = np.select([out["long_entry"], out["short_entry"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorptionReleaseParams) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for i, entry in enumerate(entries):
            if position:
                remaining -= 1
                if remaining == 0:
                    position = 0
            elif entry:
                position = int(entry)
                remaining = params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=index)
    entries = pd.DataFrame({"entry": [1, -1, -1, -1, 0, 0]}, index=index)
    for bars, expected in [(1, [0, 1, 0, -1, 0, 0]), (2, [0, 1, 1, 0, -1, -1])]:
        result = strategy.generate_signals(data, entries, None, AbsorptionReleaseParams(holding_bars=bars)).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    params = AbsorptionReleaseParams(lookback=2)
    assert strategy.indicators(data, params)["entry"].eq(0).all()
