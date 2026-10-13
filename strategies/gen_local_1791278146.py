from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecycleParams:
    territory_bars: int = 12
    pressure_bars: int = 4
    entry_threshold: float = 0.12
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[RangeRecycleParams]):
    strategy_id = "gen_local_1791278146"

    @classmethod
    def params_type(cls) -> type[RangeRecycleParams]:
        return RangeRecycleParams

    def warmup_bars(self, params: RangeRecycleParams) -> int:
        return int(params.territory_bars + params.pressure_bars)

    def indicators(self, data: pd.DataFrame, params: RangeRecycleParams) -> pd.DataFrame:
        if params.territory_bars < 1 or params.pressure_bars < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 <= params.entry_threshold < 1.0:
            raise ValueError("entry_threshold must be in [0, 1)")
        if not 0.0 < params.minimum_size <= 1.0:
            raise ValueError("minimum_size must be in (0, 1]")

        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["prior_low"] = data["low"].rolling(
            params.territory_bars, min_periods=params.territory_bars
        ).min().shift(1)
        out["prior_high"] = data["high"].rolling(
            params.territory_bars, min_periods=params.territory_bars
        ).max().shift(1)
        out["fresh_downside"] = (out["prior_low"] - data["low"]).clip(lower=0.0)
        out["fresh_upside"] = (data["high"] - out["prior_high"]).clip(lower=0.0)
        out["recycled_sell"] = (
            data["high"] - data["close"] - out["fresh_downside"]
        ).clip(lower=0.0)
        out["recycled_buy"] = (
            data["close"] - data["low"] - out["fresh_upside"]
        ).clip(lower=0.0)
        out["net_pressure"] = out["recycled_sell"] - out["recycled_buy"]
        out["pressure_sum"] = out["net_pressure"].rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum()
        out["range_sum"] = out["range"].rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum()
        out["score"] = (
            out["pressure_sum"] / out["range_sum"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRecycleParams,
    ) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan)
        # The position remains long exactly while the entry condition holds.
        # Its reversal to false is the sole exit rule.
        entry = score.notna() & score.gt(params.entry_threshold)
        strength = (
            (score - params.entry_threshold) / (1.0 - params.entry_threshold)
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["size"] = params.minimum_size + (1.0 - params.minimum_size) * strength
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeRecycleParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    close = pd.Series(100.0 + np.sin(np.arange(60)), index=index)
    data = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived["score"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    probe = pd.DataFrame({"score": [np.nan, 0.30, 0.50, 0.10, 0.40, 0.0]}, index=index[:6])
    result = strategy.generate_signals(data.iloc[:6], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert result["size"].between(params.minimum_size, 1.0).all()
    assert result["size"].iloc[3] > result["size"].iloc[2]
    prefix = strategy.indicators(data.iloc[:35], params)
    pd.testing.assert_frame_equal(prefix, derived.iloc[:35])


if __name__ == "__main__":
    _self_check()
