from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTopologyParams:
    lookback: int = 18
    edge_bias: float = 0.10
    recovery_floor: float = 1.0


class GeneratedStrategy(BaseStrategy[GapTopologyParams]):
    strategy_id = "gen_local_1791243259"

    @classmethod
    def params_type(cls) -> type[GapTopologyParams]:
        return GapTopologyParams

    @staticmethod
    def warmup_bars(params: GapTopologyParams) -> int:
        return max(4, int(params.lookback)) + 1

    def indicators(self, data: pd.DataFrame, params: GapTopologyParams) -> pd.DataFrame:
        n = max(4, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["gap_abs"] = out["gap"].abs()
        out["intraday"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        gap = out["gap"]
        magnitude = out["gap_abs"]
        positive_edges = pd.Series(0.0, index=data.index)
        negative_edges = pd.Series(0.0, index=data.index)
        all_edges = pd.Series(0.0, index=data.index)
        # Exclude adjacent edges: they exist automatically and carry no topology.
        for span in range(2, n):
            earlier_gap = gap.shift(span)
            endpoint_floor = pd.concat([magnitude, magnitude.shift(span)], axis=1).min(axis=1, skipna=False)
            interior_max = magnitude.shift(1).rolling(span - 1, min_periods=span - 1).max()
            visible = endpoint_floor.gt(interior_max)
            # Each aggregation includes only edges entirely inside the current n-bar graph.
            width = n - span
            all_edges += visible.astype(float).rolling(width, min_periods=width).sum()
            positive_edges += (visible & gap.gt(0) & earlier_gap.gt(0)).astype(float).rolling(width, min_periods=width).sum()
            negative_edges += (visible & gap.lt(0) & earlier_gap.lt(0)).astype(float).rolling(width, min_periods=width).sum()
        out["positive_edges"] = positive_edges
        out["negative_edges"] = negative_edges
        out["all_edges"] = all_edges
        out["topology_score"] = (positive_edges - negative_edges) / all_edges.replace(0.0, np.nan)
        downside = gap.lt(0)
        out["downside_gap_mass"] = (-gap).where(downside, 0.0).rolling(n, min_periods=n).sum()
        out["repair_mass"] = out["intraday"].where(downside, 0.0).rolling(n, min_periods=n).sum()
        out["recovery_score"] = out["repair_mass"] / out["downside_gap_mass"].replace(0.0, np.nan)
        out["downside_count"] = downside.astype(float).rolling(n, min_periods=n).sum()
        valid = gap.notna() & out["intraday"].notna()
        out["ready"] = valid.astype(float).rolling(n, min_periods=n).sum().eq(n)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTopologyParams) -> SignalFrame:
        topology_agrees = indicators["topology_score"].gt(params.edge_bias) & indicators["all_edges"].ge(3)
        repair_agrees = indicators["recovery_score"].gt(params.recovery_floor) & indicators["downside_count"].ge(3)
        eligible = indicators["ready"].fillna(False) & topology_agrees & repair_agrees
        df = pd.DataFrame(index=data.index)
        # Maintain the long position while both primitives agree; exit on reversal only.
        df["signal"] = eligible.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapTopologyParams()
    strategy = object.__new__(GeneratedStrategy)
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=64, freq="B")
    gaps = rng.normal(0.0, 0.006, len(index))
    intraday = rng.normal(0.0004, 0.004, len(index))
    closes = 100.0 * np.exp(np.cumsum(gaps + intraday))
    previous = np.r_[100.0, closes[:-1]]
    opens = previous * np.exp(gaps)
    data = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.001, "low": np.minimum(opens, closes) * 0.999, "close": closes, "volume": 1000000.0}, index=index)
    calculated = strategy.indicators(data, params)
    n = params.lookback
    for end in range(n, len(data)):
        window = calculated["gap"].iloc[end - n + 1:end + 1].to_numpy()
        edges = [(i, j) for i in range(n) for j in range(i + 2, n) if min(abs(window[i]), abs(window[j])) > np.max(np.abs(window[i + 1:j]))]
        assert calculated["all_edges"].iloc[end] == len(edges)
        assert calculated["positive_edges"].iloc[end] == sum(window[i] > 0 and window[j] > 0 for i, j in edges)
        assert calculated["negative_edges"].iloc[end] == sum(window[i] < 0 and window[j] < 0 for i, j in edges)
    prefix = strategy.indicators(data.iloc[:40], params)
    pd.testing.assert_frame_equal(prefix, calculated.iloc[:40])
    short = data.iloc[:5]
    gates = pd.DataFrame({"topology_score": [0.2, -0.2, 0.2, 0.2, 0.2], "all_edges": 5.0, "recovery_score": [1.2, 1.2, 0.5, 1.2, 1.2], "downside_count": 3.0, "ready": [True, True, True, True, False]}, index=short.index)
    result = strategy.generate_signals(short, gates, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= n + 1


if __name__ == "__main__":
    _self_check()
