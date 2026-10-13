from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 12
    percentile_bars: int = 126
    entry_percentile: float = 0.8

    def __post_init__(self):
        if self.graph_bars < 2 or self.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one.")


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791332564"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = out["log_close"].diff()
        n = len(data)
        upper = np.full(n, -np.inf)
        reflected_upper = np.full(n, -np.inf)
        ordinary_span = np.zeros(n)
        reflected_span = np.zeros(n)
        for lag in range(1, params.graph_bars + 1):
            slope = ((out["log_close"].shift(lag) - out["log_close"]) / lag).to_numpy()
            # A past vertex is visible exactly when its slope exceeds all nearer slopes.
            ordinary_span += lag * (slope > upper)
            reflected_span += lag * (-slope > reflected_upper)
            upper = np.fmax(upper, slope)
            reflected_upper = np.fmax(reflected_upper, -slope)
        complete = out["log_close"].rolling(params.graph_bars + 1).count().eq(params.graph_bars + 1)
        out["ordinary_span"] = pd.Series(ordinary_span, index=data.index).where(complete)
        out["reflected_span"] = pd.Series(reflected_span, index=data.index).where(complete)
        total_span = params.graph_bars * (params.graph_bars + 1) / 2.0
        out["basin_score"] = (out["reflected_span"] - out["ordinary_span"]) / total_span
        out["entry_threshold"] = out["basin_score"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        entry = (
            indicators["basin_score"].gt(indicators["entry_threshold"])
            & indicators["log_return"].lt(0)
            & indicators["entry_threshold"].notna()
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintain the long position only while the entry predicate remains true.
        df["signal"] = entry.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VisibilityParams(graph_bars=8, percentile_bars=32, entry_percentile=0.7)
    t = np.arange(180, dtype=float)
    close = np.exp(4.0 + 0.025 * np.sin(0.55 * t) + 0.005 * np.cos(1.9 * t))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(len(t), 1000000.0)},
        index=pd.date_range("2020-01-01", periods=len(t)),
    )
    strategy = GeneratedStrategy()
    ind = strategy.indicators(data, params)
    price = ind["log_close"].to_numpy()
    for end in (30, 75, 120):
        for sign, column in ((1, "ordinary_span"), (-1, "reflected_span")):
            span = 0
            for lag in range(1, params.graph_bars + 1):
                slope = sign * (price[end - lag] - price[end]) / lag
                if all(sign * price[end - k] < sign * price[end] + k * slope
                       for k in range(1, lag)):
                    span += lag
            assert ind[column].iloc[end] == span
    signals = strategy.generate_signals(data, ind, None, params).data
    predicate = ind["basin_score"].gt(ind["entry_threshold"]) & ind["log_return"].lt(0)
    expected = predicate.astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].sum() > 0
    assert set(signals["signal"].unique()) <= {0, 1}
    assert (signals["size"] > 0).all()
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    pd.testing.assert_frame_equal(ind.iloc[:120], strategy.indicators(data.iloc[:120], params))
    damaged = data.copy()
    damaged.loc[damaged.index[70], "close"] = np.nan
    damaged_ind = strategy.indicators(damaged, params)
    damaged_signals = strategy.generate_signals(damaged, damaged_ind, None, params).data
    assert damaged_signals["signal"].iloc[71] == 0


if __name__ == "__main__":
    _self_check()
