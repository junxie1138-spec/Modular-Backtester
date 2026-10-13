from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 12
    history_bars: int = 126
    percentile: float = 0.85
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791196525"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_bars + params.history_bars)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        w = params.graph_bars
        if w < 3 or params.history_bars < 10 or not 0.5 < params.percentile < 1.0:
            raise ValueError("Require graph_bars >= 3, history_bars >= 10, and 0.5 < percentile < 1.")
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        prices = np.column_stack([log_close.shift(w - 1 - k).to_numpy() for k in range(w)])
        valid = np.isfinite(prices).all(axis=1)
        n = len(data)
        upper = np.zeros((n, w, w), dtype=bool)
        lower = np.zeros_like(upper)
        # Strict natural visibility: every intervening vertex is below the connecting line.
        for left in range(w - 1):
            maximum = np.full(n, -np.inf)
            minimum = np.full(n, np.inf)
            for right in range(left + 1, w):
                slope = (prices[:, right] - prices[:, left]) / (right - left)
                upper[:, left, right] = slope > maximum
                lower[:, left, right] = slope < minimum
                maximum = np.maximum(maximum, slope)
                minimum = np.minimum(minimum, slope)
        out = pd.DataFrame(index=data.index)
        for name, graph in (("upper", upper), ("lower", lower)):
            neighbors = graph[:, :-1, -1].astype(float)
            degree = neighbors.sum(axis=1)
            wedges = degree * (degree - 1.0) / 2.0
            triangles = np.einsum("ni,nij,nj->n", neighbors, graph[:, :-1, :-1], neighbors)
            clustering = np.divide(triangles, wedges, out=np.zeros(n), where=wedges > 0)
            out[name + "_degree"] = np.where(valid, degree, np.nan)
            out[name + "_triangles"] = np.where(valid, triangles, np.nan)
            out[name + "_clustering"] = np.where(valid, clustering, np.nan)
        out["closure"] = out["upper_clustering"] - out["lower_clustering"]
        floor = close.rolling(w, min_periods=w).min()
        ceiling = close.rolling(w, min_periods=w).max()
        out["range_floor"] = floor
        out["range_ceiling"] = ceiling
        out["position"] = (close - floor) / (ceiling - floor).replace(0.0, np.nan)
        for column in ("closure", "position"):
            baseline = out[column].shift(1).rolling(params.history_bars, min_periods=params.history_bars)
            out[column + "_low"] = baseline.quantile(1.0 - params.percentile)
            out[column + "_high"] = baseline.quantile(params.percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold not in (3, 4, 5):
            raise ValueError("Require a positive profit_target and max_hold of 3, 4, or 5.")
        cols = ["closure", "position", "closure_low", "closure_high", "position_low", "position_high"]
        ready = np.isfinite(indicators[cols]).all(axis=1)
        buy = ready & (indicators["closure"] < indicators["closure_low"]) & (indicators["position"] <= indicators["position_low"])
        sell = ready & (indicators["closure"] > indicators["closure_high"]) & (indicators["position"] >= indicators["position_high"])
        candidate = np.select([buy, sell], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry = np.nan
        for i in range(len(data)):
            if side:
                # The previous close's instruction fills at this bar's open.
                if i == entry_bar:
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                gain = side * (closes[i] / entry - 1.0)
                expired = i - entry_bar + 1 >= params.max_hold
                if expired or (np.isfinite(gain) and gain >= params.profit_target):
                    side = 0
                    continue
            elif candidate[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidate[i])
                entry_bar = i + 1
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0,
                         "close": [100.0, 100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                         "volume": 1000.0}, index=index)
    features = pd.DataFrame({"closure": [-2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             "position": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5],
                             "closure_low": -1.0, "closure_high": 1.0,
                             "position_low": 0.2, "position_high": 0.8}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, features, None, VisibilityParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    small = VisibilityParams(graph_bars=3, history_bars=10)
    curve = pd.DataFrame({"close": np.exp(np.arange(20, dtype=float) ** 2 / 1000.0)})
    topology = strategy.indicators(curve, small)
    assert topology["upper_clustering"].iloc[-1] == 1.0
    assert topology["lower_clustering"].iloc[-1] == 0.0
    assert strategy.warmup_bars(small) == 13


if __name__ == "__main__":
    _self_check()
