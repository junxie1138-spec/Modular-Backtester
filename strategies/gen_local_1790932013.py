from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRejectionParams:
    lookback: int = 20
    min_extension: float = 0.12
    min_overlap: float = 0.65
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeRejectionParams]):
    strategy_id = "gen_local_1790932013"

    @classmethod
    def params_type(cls):
        return RangeRejectionParams

    def warmup_bars(self, params: RangeRejectionParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RangeRejectionParams) -> pd.DataFrame:
        if params.lookback < 2 or params.hold_bars < 1:
            raise ValueError("lookback must be >= 2 and hold_bars >= 1")
        if params.min_extension <= 0 or not 0 < params.min_overlap <= 1:
            raise ValueError("Invalid range thresholds")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["baseline_range"] = out["range"].rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["prior_midpoint"] = (out["prior_high"] + out["prior_low"]) / 2.0
        out["prior_range"] = out["range"].shift(1)
        upper = pd.concat([data["high"], out["prior_high"]], axis=1).min(axis=1, skipna=False)
        lower = pd.concat([data["low"], out["prior_low"]], axis=1).max(axis=1, skipna=False)
        out["overlap"] = (upper - lower).clip(lower=0) / out["prior_range"]
        out["down_extension"] = (out["prior_low"] - data["low"]) / out["baseline_range"]
        out["up_extension"] = (data["high"] - out["prior_high"]) / out["baseline_range"]
        valid = out[["range", "baseline_range", "prior_range", "overlap"]].notna().all(axis=1)
        overlap = valid & out["overlap"].ge(params.min_overlap)
        long_rejection = overlap & out["down_extension"].ge(params.min_extension) & out["up_extension"].le(0) & data["close"].ge(out["prior_midpoint"])
        short_rejection = overlap & out["up_extension"].ge(params.min_extension) & out["down_extension"].le(0) & data["close"].le(out["prior_midpoint"])
        out["rejection_side"] = np.select([long_rejection, short_rejection], [1, -1], default=0).astype(int)
        out["entry_side"] = out["rejection_side"].where(out["rejection_side"].eq(out["rejection_side"].shift(1)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeRejectionParams) -> SignalFrame:
        entries = indicators["entry_side"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, entry in enumerate(entries):
            if position != 0 and i - entry_bar >= params.hold_bars:
                position = 0
                continue
            if position == 0 and entry != 0:
                position = int(entry)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeRejectionParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    assert not derived["entry_side"].any()
    entries = pd.DataFrame({"entry_side": np.zeros(len(data), dtype=int)}, index=index)
    entries.iloc[[1, 3, 11, 13], 0] = [1, -1, -1, 1]
    result = strategy.generate_signals(data, entries, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[2:10] = 1
    expected[12:20] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(data.index) and result["size"].gt(0).all()
