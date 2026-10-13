from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTerritoryParams:
    lookback: int = 4
    threshold: float = 0.20

    def __post_init__(self):
        if not 2 <= self.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")


class GeneratedStrategy(BaseStrategy[RangeTerritoryParams]):
    strategy_id = "gen_local_1790918558"

    @classmethod
    def params_type(cls):
        return RangeTerritoryParams

    def warmup_bars(self, params: RangeTerritoryParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: RangeTerritoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["bar_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["upper_new"] = (
            data["high"] - np.maximum(data["low"], out["previous_high"])
        ).clip(lower=0)
        out["lower_new"] = (
            np.minimum(data["high"], out["previous_low"]) - data["low"]
        ).clip(lower=0)
        out["net_new"] = out["upper_new"] - out["lower_new"]
        out["territory_sum"] = out["net_new"].rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out["range_sum"] = out["bar_range"].rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out["pressure"] = out["territory_sum"] / out["range_sum"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTerritoryParams,
    ) -> SignalFrame:
        pressure = indicators["pressure"].fillna(0.0)
        intent = np.where(
            pressure >= params.threshold, 1,
            np.where(pressure <= -params.threshold, -1, 0),
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(intent):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeTerritoryParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "open": np.arange(8, dtype=float) + 10,
        "high": np.arange(8, dtype=float) + 12,
        "low": np.arange(8, dtype=float) + 10,
        "close": np.arange(8, dtype=float) + 11,
        "volume": 1000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert np.isfinite(derived.to_numpy()).all()
    assert derived["pressure"].iloc[:params.lookback].eq(0).all()
    assert derived["pressure"].iloc[params.lookback:].eq(0.5).all()
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    forced = pd.DataFrame({"pressure": [0.3, 0, -0.3, -0.3, 0, 0.3, 0.3, 0]}, index=index)
    signals = strategy.generate_signals(data, forced, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert strategy.warmup_bars(RangeTerritoryParams(lookback=9)) == 10


if __name__ == "__main__":
    _self_check()
