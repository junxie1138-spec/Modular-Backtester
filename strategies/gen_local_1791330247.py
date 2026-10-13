from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 4
    percentile_bars: int = 60
    topology_percentile: float = 0.60
    compression_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791330247"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_bars < 3 or params.percentile_bars < 3:
            raise ValueError("Lookbacks must be at least three bars")
        if not 0.5 < params.topology_percentile < 1.0:
            raise ValueError("Topology percentile must be between 0.5 and 1")
        if not 0.0 < params.compression_percentile < 1.0:
            raise ValueError("Compression percentile must be between 0 and 1")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        # ponytail: O(graph_bars squared) vector passes; use hull algorithms if windows grow large.
        for lag in range(1, params.graph_bars):
            past = close.shift(lag)
            upper_edge = close.notna() & past.notna()
            lower_edge = upper_edge.copy()
            for offset in range(1, lag):
                intermediate = close.shift(offset)
                chord = close + (past - close) * (offset / lag)
                upper_edge &= intermediate < chord
                lower_edge &= intermediate > chord
            upper_degree += upper_edge.astype(float)
            lower_degree += lower_edge.astype(float)
        complete = close.rolling(params.graph_bars, min_periods=params.graph_bars).count().eq(params.graph_bars)
        upper_degree = upper_degree.where(complete)
        lower_degree = lower_degree.where(complete)
        score = lower_degree - upper_degree
        high = data["high"].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data["low"].astype(float).replace([np.inf, -np.inf], np.nan)
        relative_range = ((high - low) / close).where(high >= low)
        range_mean = relative_range.rolling(2, min_periods=2).mean()
        prior_score = score.shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        prior_range = range_mean.shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        return pd.DataFrame({
            "upper_degree": upper_degree,
            "lower_degree": lower_degree,
            "score": score,
            "relative_range": relative_range,
            "range_mean": range_mean,
            "upper_threshold": prior_score.quantile(params.topology_percentile),
            "lower_threshold": prior_score.quantile(1.0 - params.topology_percentile),
            "compression_threshold": prior_range.quantile(params.compression_percentile),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        ready = indicators[["score", "range_mean", "upper_threshold", "lower_threshold", "compression_threshold"]].notna().all(axis=1)
        compressed = ready & indicators["range_mean"].le(indicators["compression_threshold"])
        long_entry = compressed & indicators["score"].gt(0) & indicators["score"].ge(indicators["upper_threshold"])
        short_entry = compressed & indicators["score"].lt(0) & indicators["score"].le(indicators["lower_threshold"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "score": [2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        "range_mean": 0.01,
        "compression_threshold": 0.02,
        "upper_threshold": 1.0,
        "lower_threshold": -1.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result.data["size"].gt(0).all()
    missing = indicators.copy()
    missing["score"] = np.nan
    assert strategy.generate_signals(data, missing, None, params).data["signal"].eq(0).all()
    curve_index = pd.date_range("2020-01-01", periods=20)
    close = 100.0 + np.arange(20, dtype=float) ** 2
    curve = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=curve_index)
    topology = strategy.indicators(curve, params)
    assert topology["upper_degree"].iloc[3:].eq(3).all()
    assert topology["lower_degree"].iloc[3:].eq(1).all()
    assert topology.index.equals(curve.index)


if __name__ == "__main__":
    _self_check()
