from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    smoothing_window: int = 20
    percentile_window: int = 252
    entry_percentile: float = 0.70


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791277427"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_window + params.smoothing_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if min(params.graph_window, params.smoothing_window, params.percentile_window) < 2:
            raise ValueError("All windows must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        adjacent = close.notna() & previous.notna()
        trough_degree = adjacent.astype(float)
        peak_degree = adjacent.astype(float)
        interior_min = previous.copy()
        interior_max = previous.copy()

        # Horizontal visibility: both endpoints must clear every interior bar.
        for lag in range(2, params.graph_window + 1):
            past = close.shift(lag)
            trough_degree += ((close < interior_min) & (past < interior_min)).astype(float)
            peak_degree += ((close > interior_max) & (past > interior_max)).astype(float)
            interior_min = pd.Series(np.minimum(interior_min.to_numpy(), past.to_numpy()), index=data.index)
            interior_max = pd.Series(np.maximum(interior_max.to_numpy(), past.to_numpy()), index=data.index)

        complete = close.rolling(params.graph_window + 1, min_periods=1).count().eq(params.graph_window + 1)
        trough_degree = trough_degree.where(complete)
        peak_degree = peak_degree.where(complete)
        imbalance = (trough_degree - peak_degree) / float(params.graph_window)
        pressure = imbalance.rolling(params.smoothing_window, min_periods=params.smoothing_window).mean()
        threshold = pressure.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)

        return pd.DataFrame({
            "trough_degree": trough_degree,
            "peak_degree": peak_degree,
            "connectivity_imbalance": imbalance,
            "topology_pressure": pressure,
            "entry_threshold": threshold,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        pressure = indicators["topology_pressure"]
        threshold = indicators["entry_threshold"]
        entry_condition = pressure.notna() & threshold.notna() & pressure.gt(threshold)
        df = pd.DataFrame(index=data.index)
        # The position exits precisely when the entry condition becomes false.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VisibilityParams(graph_window=2, smoothing_window=2, percentile_window=3)
    close = pd.Series(100.0 + np.tile([3, 5, 2, 6, 1, 7, 2, 6], 5), index=pd.date_range("2020-01-01", periods=40))
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    expected = indicators["topology_pressure"].gt(indicators["entry_threshold"]).astype(int).shift(1).fillna(0).astype(int)
    assert indicators["trough_degree"].iloc[2] == 2.0
    assert indicators["peak_degree"].iloc[2] == 1.0
    assert signals["signal"].equals(expected)
    assert signals["signal"].isin([0, 1]).all() and signals["signal"].eq(1).any()
    assert signals["size"].gt(0).all()
    first_valid = indicators["entry_threshold"].first_valid_index()
    assert first_valid is not None and data.index.get_loc(first_valid) < strategy.warmup_bars(params)
