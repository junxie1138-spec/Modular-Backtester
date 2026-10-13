from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_bars: int = 8
    rank_bars: int = 100
    entry_quantile: float = 0.85
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791229025"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params):
        return int(max(params.graph_bars + params.rank_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        if (params.graph_bars < 3 or params.rank_bars < 2 or params.atr_bars < 1
                or not 0 < params.entry_quantile < 1 or params.breakeven_pct <= 0
                or params.trail_atr <= 0 or not 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid topology or exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        out["activity"] = out["return"].abs()
        activity = out["activity"]
        edges = {}
        minimum_slope = pd.Series(np.inf, index=data.index)
        # Natural visibility: intermediates lie below the connecting line.
        for distance in range(1, params.graph_bars):
            slope = (activity - activity.shift(distance)) / distance
            edge = slope.notna() & (slope < minimum_slope)
            edges[distance] = edge
            out[f"edge_{distance}"] = edge.astype(int)
            minimum_slope = pd.concat([minimum_slope, slope], axis=1).min(axis=1)
        out["degree"] = sum(edge.astype(int) for edge in edges.values())
        triangles = pd.Series(0.0, index=data.index)
        # ponytail: O(graph_bars squared); use compiled motifs if windows become large.
        for near in range(1, params.graph_bars - 1):
            for far in range(near + 1, params.graph_bars):
                linked = edges[far - near].shift(near, fill_value=False)
                triangles += (edges[near] & edges[far] & linked).astype(int)
        out["triangles"] = triangles
        pairs = out["degree"] * (out["degree"] - 1) / 2.0
        out["cohesion"] = (triangles / pairs.replace(0, np.nan)).fillna(0.0)
        out["net_move"] = out["return"].rolling(params.graph_bars).sum()
        out["gross_move"] = activity.rolling(params.graph_bars).sum()
        out["efficiency"] = (out["net_move"].abs()
                             / out["gross_move"].replace(0, np.nan)).fillna(0.0)
        out["direction"] = np.sign(out["net_move"].fillna(0.0))
        out["score"] = (out["cohesion"] * out["efficiency"]).where(out["net_move"].notna())
        out["threshold"] = out["score"].shift(1).rolling(params.rank_bars).quantile(params.entry_quantile)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        eligible = ((indicators["score"] > indicators["threshold"])
                    & indicators["threshold"].notna()
                    & indicators["atr"].gt(0))
        candidates = np.where(eligible, indicators["direction"], 0).astype(int)
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        side, age = 0, 0
        entry, stop, peak = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            if side:
                if np.isnan(entry):
                    # The preceding decision fills at this bar's open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    peak = side * entry
                    stop = peak - params.trail_atr * atrs[i - 1]
                age += 1
                price, atr = closes[i], atrs[i]
                if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
                    side = 0
                    continue
                favorable = side * price
                # Stops are evaluated at the close and executed next bar.
                if favorable <= stop or age >= params.hold_bars:
                    side = 0
                    continue
                peak = max(peak, favorable)
                if (favorable - side * entry) / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stop only rises; a short's actual stop falls.
                    stop = max(stop, side * entry, peak - params.trail_atr * atr)
                raw[i] = side
            elif candidates[i] and i >= self.warmup_bars(params) - 1:
                side = int(candidates[i])
                entry, stop, peak = np.nan, np.nan, np.nan
                age, armed = 0, False
                raw[i] = side
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = TopologyParams(graph_bars=3, rank_bars=2, atr_bars=2,
                            breakeven_pct=0.005, trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=12)
    prices = np.full(12, 100.0)
    prices[6:9] = [101.0, 103.0, 101.5]
    data = pd.DataFrame({"open": 100.0, "high": prices + 1,
                         "low": prices - 1, "close": prices, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"score": 0.0, "threshold": 0.5,
                              "direction": 1.0, "atr": 1.0}, index=index)
    features.loc[index[5], "score"] = 1.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    data["close"] = 200.0 - data["close"]
    features["direction"] = -1.0
    mirrored = strategy.generate_signals(data, features, ctx, params).data
    assert mirrored["signal"].tolist() == (-result["signal"]).tolist()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert np.isfinite(actual["score"].dropna()).all()
    assert strategy.indicators(data.iloc[:8], params).equals(actual.iloc[:8])
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
