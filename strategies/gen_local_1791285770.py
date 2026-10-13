from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 20
    flow_window: int = 5
    degree_min: int = 4
    degree_margin: int = 2
    gap_floor: float = 0.002
    flow_threshold: float = 0.12
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791285770"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.graph_window + 1, params.flow_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_window < 3 or params.flow_window < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; graph_window >= 3")
        if params.degree_min < 1 or params.degree_margin < 1:
            raise ValueError("Degree thresholds must be positive")
        if not (params.gap_floor >= 0 and 0 <= params.flow_threshold < 1):
            raise ValueError("Invalid gap or pressure threshold")
        previous = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"].where(lambda s: s > 0) / previous - 1).replace([np.inf, -np.inf], np.nan)
        current = gap.to_numpy(dtype=float)
        upper_blocker = np.full(len(data), -np.inf)
        lower_blocker = np.full(len(data), -np.inf)
        upper_degree = np.zeros(len(data), dtype=float)
        lower_degree = np.zeros(len(data), dtype=float)
        # Horizontal visibility: every intervening node is below both endpoints.
        for lag in range(1, params.graph_window):
            older = gap.shift(lag).to_numpy(dtype=float)
            upper_degree += np.minimum(current, older) > upper_blocker
            lower_degree += np.minimum(-current, -older) > lower_blocker
            upper_blocker = np.maximum(upper_blocker, older)
            lower_blocker = np.maximum(lower_blocker, -older)
        ready = gap.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        span = (data["high"] - data["low"]).where(lambda s: s > 0)
        body = ((data["close"] - data["open"]) / span).clip(-1, 1)
        volume = data["volume"].where(lambda s: s >= 0).replace([np.inf, -np.inf], np.nan)
        weighted_body = (body * volume).rolling(params.flow_window, min_periods=params.flow_window).sum()
        volume_sum = volume.rolling(params.flow_window, min_periods=params.flow_window).sum()
        pressure = (weighted_body / volume_sum.where(lambda s: s > 0)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "gap": gap,
            "upper_degree": pd.Series(upper_degree, index=data.index).where(ready),
            "lower_degree": pd.Series(lower_degree, index=data.index).where(ready),
            "pressure": pressure,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        g = indicators
        valid = np.isfinite(g[["gap", "upper_degree", "lower_degree", "pressure"]]).all(axis=1)
        long_graph = (g["gap"] > params.gap_floor) & (g["upper_degree"] >= params.degree_min) & ((g["upper_degree"] - g["lower_degree"]) >= params.degree_margin)
        short_graph = (g["gap"] < -params.gap_floor) & (g["lower_degree"] >= params.degree_min) & ((g["lower_degree"] - g["upper_degree"]) >= params.degree_margin)
        long_entry = valid & long_graph & (g["pressure"] > params.flow_threshold)
        short_entry = valid & short_graph & (g["pressure"] < -params.flow_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position, entered = 0, -1
        for i, entry in enumerate(entries):
            if position:
                if i - entered == params.hold_bars:
                    position = 0
                # Entry votes never close, reverse, or extend an active trade.
            elif entry:
                position, entered = int(entry), i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        # Shift reference: https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=32)
    close = pd.Series(100.0 + np.arange(32) * 0.1, index=index)
    data = pd.DataFrame({"open": close - 0.05, "high": close + 0.2,
                         "low": close - 0.2, "close": close, "volume": 1000.0})
    strategy, params = GeneratedStrategy(), Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:26], params)
    assert prefix.equals(full.iloc[:26])
    assert full["upper_degree"].iloc[:params.graph_window].isna().all()
    g = pd.DataFrame({"gap": 0.01, "upper_degree": 8.0,
                      "lower_degree": 1.0, "pressure": 0.0}, index=index)
    g.loc[index[2:12], "pressure"] = 1.0
    g.loc[index[12:], ["gap", "upper_degree", "lower_degree", "pressure"]] = [-0.01, 1.0, 8.0, -1.0]
    signal = strategy.generate_signals(data, g, ctx, params).data["signal"].to_numpy()
    assert np.all(signal[:3] == 0) and np.all(signal[3:11] == 1)
    assert signal[11] == 0 and signal[12] == 1
    assert np.all(signal[12:20] == 1) and signal[20] == 0
    assert np.all(signal[21:29] == -1) and signal[29] == 0
    g["pressure"] = -1.0
    g["gap"], g["upper_degree"], g["lower_degree"] = 0.01, 8.0, 1.0
    assert strategy.generate_signals(data, g, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
