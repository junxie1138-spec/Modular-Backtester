from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    horizon: int = 18
    overlap_threshold: float = 0.60
    frontier_threshold: float = 0.15

    def __post_init__(self):
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 0.0 < self.overlap_threshold < 1.0:
            raise ValueError("overlap_threshold must be between zero and one")
        if not 0.0 < self.frontier_threshold < 1.0:
            raise ValueError("frontier_threshold must be between zero and one")


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local-w2_1791305904"

    @classmethod
    def params_type(cls):
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return int(max(200, params.horizon + 1))

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high, low = data["high"], data["low"]
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        pair_high = pd.concat([high, high.shift(1)], axis=1)
        pair_low = pd.concat([low, low.shift(1)], axis=1)
        out["intersection"] = (
            pair_high.min(axis=1, skipna=False)
            - pair_low.max(axis=1, skipna=False)
        ).clip(lower=0.0)
        out["union"] = (
            pair_high.max(axis=1, skipna=False)
            - pair_low.min(axis=1, skipna=False)
        )
        out["overlap"] = out["intersection"] / out["union"].where(out["union"] > 0)
        denominator = data["close"].where(data["close"] > 0)
        out["upper_extension"] = high.diff().clip(lower=0.0) / denominator
        out["lower_extension"] = (-low.diff()).clip(lower=0.0) / denominator
        window = params.horizon
        out["mean_overlap"] = out["overlap"].rolling(window, min_periods=window).mean()
        out["upper_capacity"] = out["upper_extension"].rolling(window, min_periods=window).sum()
        out["lower_capacity"] = out["lower_extension"].rolling(window, min_periods=window).sum()
        out["total_capacity"] = out["upper_capacity"] + out["lower_capacity"]
        out["frontier_balance"] = (
            (out["upper_capacity"] - out["lower_capacity"])
            / out["total_capacity"].where(out["total_capacity"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeParams,
    ) -> SignalFrame:
        crowded = indicators["mean_overlap"] >= params.overlap_threshold
        migrating = indicators["mean_overlap"] < params.overlap_threshold
        balance = indicators["frontier_balance"]
        reversal_setup = crowded & (balance < -params.frontier_threshold)
        continuation_setup = migrating & (balance > params.frontier_threshold)
        ready = indicators[["ma200", "mean_overlap", "frontier_balance"]].notna().all(axis=1)
        entry_condition = (
            ready
            & (data["close"] > indicators["ma200"])
            & (reversal_setup | continuation_setup)
        )
        # The position remains long exactly while the entry condition is true.
        # Its first false value is the only exit; horizon is signal memory.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeParams()
    index = pd.date_range("2020-01-01", periods=260, freq="B")
    close = pd.Series(100.0 + np.arange(260) * 0.1, index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:240], strategy.indicators(data.iloc[:240], params))
    sample = data.iloc[:6]
    setups = pd.DataFrame({
        "ma200": [90.0, 90.0, 90.0, 90.0, 90.0, 110.0],
        "mean_overlap": [0.8, 0.8, 0.2, 0.2, 0.8, 0.8],
        "frontier_balance": [-0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
    }, index=sample.index)
    result = strategy.generate_signals(sample, setups, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 0, 1]
    assert result["size"].gt(0).all()
    actual = strategy.generate_signals(data, full, ctx, params).data
    assert actual["signal"].iloc[:200].eq(0).all()
    assert set(actual["signal"].unique()) <= {0, 1}
