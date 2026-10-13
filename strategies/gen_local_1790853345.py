from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalCapacityParams:
    fractional_order: float = 0.6
    capacity_threshold: float = 0.35
    target_daily_vol: float = 0.008
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FractionalCapacityParams]):
    strategy_id = "gen_local_1790853345"

    @classmethod
    def params_type(cls) -> type[FractionalCapacityParams]:
        return FractionalCapacityParams

    def warmup_bars(self, params: FractionalCapacityParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: FractionalCapacityParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.capacity_threshold) or params.capacity_threshold <= 0.0:
            raise ValueError("capacity_threshold must be finite and positive")
        if not np.isfinite(params.target_daily_vol) or params.target_daily_vol <= 0.0:
            raise ValueError("target_daily_vol must be finite and positive")
        if not 0.0 < params.max_size <= 1.0:
            raise ValueError("max_size must be between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: x > 0.0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["daily_vol"] = out["return"].rolling(5, min_periods=5).std(ddof=0)

        weights = [1.0]
        for lag in range(1, 4):
            weights.append(-weights[-1] * (params.fractional_order - lag + 1) / lag)
        # ponytail: four-tap memory; lengthen the filter only if the warmup limit changes.
        out["fractional_close"] = sum(weights[lag] * close.shift(lag) for lag in range(4))
        out["fractional_close"] -= sum(weights) * close.shift(3)
        out["pressure"] = out["fractional_close"] / close / out["daily_vol"].clip(lower=0.002)
        out["pressure"] = out["pressure"].replace([np.inf, -np.inf], np.nan)
        out["pressure_change"] = out["pressure"].diff()
        out["target_size"] = (params.target_daily_vol / out["daily_vol"].clip(lower=0.002)).clip(upper=params.max_size)
        out["target_size"] = out["target_size"].replace([np.inf, -np.inf], np.nan).fillna(params.max_size)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalCapacityParams,
    ) -> SignalFrame:
        pressure = indicators["pressure"]
        change = indicators["pressure_change"]
        ready = np.isfinite(pressure) & np.isfinite(change)
        long_entry = ready & (pressure <= -params.capacity_threshold) & (change > 0.0)
        short_entry = ready & (pressure >= params.capacity_threshold) & (change < 0.0)
        entry = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)

        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(entry):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = FractionalCapacityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    pressure = pd.Series([1.0, 0.7, 0.5, -1.0, -0.7, -0.5, 1.0, 0.7], index=index)
    indicators = pd.DataFrame({"pressure": pressure, "pressure_change": pressure.diff(), "target_size": 0.5}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, -1, 0, 1, 1]
    assert result["size"].gt(0.0).all()
    assert result["size"].iloc[0] == params.max_size
    assert result["size"].iloc[1] == 0.5
