from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    memory_bars: int = 32
    hold_bars: int = 7

    def __post_init__(self):
        if self.memory_bars < 2 or self.hold_bars < 1:
            raise ValueError("memory_bars must be >= 2 and hold_bars must be >= 1")


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local-w2_1791268966"

    @classmethod
    def params_type(cls):
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(params.memory_bars + 26)

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        weights = np.empty(params.memory_bars, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.memory_bars):
            weights[k] = weights[k - 1] * (k - 1 - 0.4) / k
        # ponytail: finite fractional-memory filter; lengthen memory_bars if truncation matters.
        fractional = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional = fractional + weight * out["log_close"].shift(k)
        out["fractional_close"] = fractional
        out["innovation"] = fractional.diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["gap"] = data["open"].where(data["open"] > 0) / close.shift(1) - 1.0
        out["return"] = close.pct_change(fill_method=None)
        out["return_volatility"] = out["return"].rolling(20, min_periods=20).std()
        out["gap_response"] = out["gap"].rolling(20, min_periods=20).corr(out["lagged_innovation"])
        out["prior_gap_response"] = out["gap_response"].shift(5)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["gap_response"] < 0.0)
            & (out["prior_gap_response"] >= 0.0)
            & (out["innovation"] > 0.0)
            & (out["return_volatility"] > 0.0)
            & (out["gap"] < -0.25 * out["return_volatility"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar >= params.hold_bars:
                    entry_bar = -1
            elif entries[i]:
                entry_bar = i
            signals[i] = int(entry_bar >= 0)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapMemoryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=2 * params.hold_bars + 2, freq="B")
    data = pd.DataFrame(index=index)
    entries = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[1:params.hold_bars + 1] = 1
    expected[params.hold_bars + 2:2 * params.hold_bars + 2] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
