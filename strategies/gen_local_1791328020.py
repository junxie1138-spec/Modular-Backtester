from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 12
    baseline_window: int = 20
    topology_threshold: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791328020"

    @classmethod
    def params_type(cls) -> type[TopologyParams]:
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(params.graph_window + params.baseline_window)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.baseline_window < 2:
            raise ValueError("Graph window must be >= 3 and baseline window >= 2")
        if not np.isfinite(params.topology_threshold) or params.topology_threshold <= 0:
            raise ValueError("Topology threshold must be finite and positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")

        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = upper_degree.copy()
        intervening_high = pd.Series(-np.inf, index=data.index)
        intervening_low = pd.Series(np.inf, index=data.index)

        # Horizontal visibility requires every intervening close to lie below
        # both endpoints; the inverted graph reverses this inequality.
        for lag in range(1, params.graph_window):
            if lag > 1:
                middle = close.shift(lag - 1)
                intervening_high = np.maximum(intervening_high, middle)
                intervening_low = np.minimum(intervening_low, middle)
            previous = close.shift(lag)
            upper_degree += ((close > intervening_high) & (previous > intervening_high)).astype(float)
            lower_degree += ((close < intervening_low) & (previous < intervening_low)).astype(float)

        valid = close.rolling(params.graph_window, min_periods=1).count().eq(params.graph_window)
        upper_degree = upper_degree.where(valid)
        lower_degree = lower_degree.where(valid)
        asymmetry = (upper_degree - lower_degree) / (params.graph_window - 2)
        baseline = asymmetry.shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        return pd.DataFrame({
            "upper_degree": upper_degree,
            "lower_degree": lower_degree,
            "asymmetry": asymmetry,
            "baseline": baseline,
            "topology_change": asymmetry - baseline,
            "rising_close": close.gt(close.shift(1)),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        candidate = (
            indicators["topology_change"].ge(params.topology_threshold)
            & indicators["rising_close"].fillna(False).astype(bool)
        ).fillna(False)
        confirmed = candidate & candidate.shift(1, fill_value=False)
        entry = confirmed & ~confirmed.shift(1, fill_value=False)

        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entry.to_numpy(dtype=bool)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    sample = pd.DataFrame({"close": np.arange(100.0, 108.0)}, index=index)
    derived = pd.DataFrame({
        "topology_change": [0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
        "rising_close": True,
    }, index=index)
    for hold in (1, 2):
        result = GeneratedStrategy.generate_signals(
            None, sample, derived, None, TopologyParams(hold_bars=hold)
        ).data["signal"]
        expected = np.zeros(8, dtype=int)
        expected[3:3 + hold] = 1
        assert np.array_equal(result.to_numpy(), expected)
