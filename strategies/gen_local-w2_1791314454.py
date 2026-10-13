from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_bars: int = 12
    topology_jump: float = 0.25
    displacement_atr: float = 0.6
    atr_bars: int = 14
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791314454"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params):
        return int(max(params.graph_bars + 2, params.atr_bars + 1, 4))

    def indicators(self, data, params):
        if (params.graph_bars < 2 or params.atr_bars < 2
                or not 3 <= params.hold_bars <= 5
                or not all(np.isfinite(v) and v > 0 for v in
                           (params.topology_jump, params.displacement_atr,
                            params.trail_atr))):
            raise ValueError("Invalid topology, ATR, threshold or holding parameters")
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = upper_degree.copy()
        min_slope = pd.Series(np.inf, index=data.index)
        max_slope = pd.Series(-np.inf, index=data.index)
        # Exact backward natural visibility: an edge clears every nearer node.
        for lag in range(1, params.graph_bars + 1):
            slope = (log_close - log_close.shift(lag)) / lag
            upper_degree += (slope < min_slope).astype(float)
            lower_degree += (slope > max_slope).astype(float)
            min_slope = np.minimum(min_slope, slope)
            max_slope = np.maximum(max_slope, slope)
        valid = log_close.rolling(params.graph_bars + 1).count().eq(params.graph_bars + 1)
        upper_degree = upper_degree.where(valid)
        lower_degree = lower_degree.where(valid)
        balance = (upper_degree - lower_degree) / params.graph_bars
        previous_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        displacement = (close - close.shift(3)) / atr.where(atr > 0)
        return pd.DataFrame({
            "log_close": log_close,
            "upper_degree": upper_degree,
            "lower_degree": lower_degree,
            "balance": balance,
            "topology_change": balance.diff(),
            "true_range": tr,
            "atr": atr,
            "displacement": displacement,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        topology = indicators["topology_change"]
        displacement = indicators["displacement"]
        long_entry = topology.ge(params.topology_jump) & displacement.ge(params.displacement_atr)
        short_entry = topology.le(-params.topology_jump) & displacement.le(-params.displacement_atr)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = age = 0
        high_water = -np.inf
        low_water = np.inf
        stop = -np.inf
        cover_stop = np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position = age = 0
                continue
            if position:
                # Entry fills today; its decision-bar close is excluded.
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                low_water = min(low_water, close)
                cover_stop = min(cover_stop, low_water + params.trail_atr * atr)
                # Required high-water drawdown exit applies to both directions.
                # Shorts also have a mirrored adverse-move trailing exit.
                if close <= stop or (position < 0 and close >= cover_stop) or age >= params.hold_bars:
                    position = age = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water, low_water = -np.inf, np.inf
                stop, cover_stop = -np.inf, np.inf
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Small runnable check: AND gate, one-bar lag, ratchet and time exit.
    strategy = GeneratedStrategy()
    params = TopologyParams(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 100., 103., 101., 101., 101., 101.]}, index=index)
    indicators = pd.DataFrame({
        "topology_change": [0.5, 0., 0., 0., 0., 0., 0.],
        "displacement": [1., 0., 0., 0., 0., 0., 0.],
        "atr": [1., 1., 1., 10., 1., 1., 1.],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=16)
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    indicators["displacement"] = 0.
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    indicators.loc[index[0], "displacement"] = 1.
    data["close"] = 100.
    indicators["atr"] = 1.
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators.loc[index[0], ["topology_change", "displacement"]] = [-0.5, -1.]
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, -1, 0, 0]
