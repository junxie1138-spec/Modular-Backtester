from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapScarcityParams:
    baseline_window: int = 20
    gap_multiple: float = 1.5
    volume_ceiling: float = 0.8

    def __post_init__(self):
        if self.baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if not np.isfinite(self.gap_multiple) or self.gap_multiple <= 0:
            raise ValueError("gap_multiple must be positive and finite")
        if not np.isfinite(self.volume_ceiling) or not 0 < self.volume_ceiling < 1:
            raise ValueError("volume_ceiling must lie strictly between 0 and 1")


class GeneratedStrategy(BaseStrategy[GapScarcityParams]):
    strategy_id = "gen_local_1791193967"

    @classmethod
    def params_type(cls):
        return GapScarcityParams

    def warmup_bars(self, params: GapScarcityParams) -> int:
        return int(params.baseline_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapScarcityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["prior_close"].where(out["prior_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        window = params.baseline_window
        out["gap_capacity"] = out["gap"].abs().rolling(window, min_periods=window).median().shift(1)
        out["volume_capacity"] = data["volume"].rolling(window, min_periods=window).median().shift(1)
        out["gap_strength"] = out["gap"] / out["gap_capacity"].where(out["gap_capacity"] > 0)
        out["volume_ratio"] = data["volume"] / out["volume_capacity"].where(out["volume_capacity"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapScarcityParams) -> SignalFrame:
        valid = np.isfinite(indicators["gap_strength"]) & np.isfinite(indicators["volume_ratio"])
        gap_agrees = indicators["gap_strength"] >= params.gap_multiple
        scarcity_agrees = indicators["volume_ratio"].gt(0) & indicators["volume_ratio"].le(params.volume_ceiling)
        df = pd.DataFrame(index=data.index)
        # The joint predicate is the position target; either predicate flipping exits.
        # No expiry: repeated agreement can extend the intended 1-2 session holding period.
        df["signal"] = (valid & gap_agrees & scarcity_agrees).fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapScarcityParams(baseline_window=3)
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    opening = np.array([100.0, 100.1, 99.9, 100.1, 100.3, 100.4, 99.9, 100.1, 100.1])
    data = pd.DataFrame({"open": opening, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": [100.0, 100.0, 100.0, 100.0, 60.0, 60.0, 100.0, 100.0, 100.0]}, index=index)
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert result.index.equals(data.index) and result["size"].gt(0).all()
    prefix = strategy.indicators(data.iloc[:6], params)
    pd.testing.assert_frame_equal(indicators.iloc[:6], prefix)
    assert strategy.warmup_bars(params) == 5


if __name__ == "__main__":
    _self_check()
