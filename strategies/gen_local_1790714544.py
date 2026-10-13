from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 18
    position_window: int = 20
    z_window: int = 60
    degree_z: float = 1.0
    edge_fraction: float = 0.25


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790714544"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @classmethod
    def warmup_bars(cls, params: VisibilityParams) -> int:
        return max(params.graph_window, params.position_window, params.z_window) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)
        degree = np.full(n, np.nan, dtype=float)
        w = params.graph_window
        for i in range(w - 1, n):
            start = i - w + 1
            visible_count = 0
            for j in range(start, i):
                span = i - j
                if span == 1:
                    visible_count += 1
                    continue
                left = close[j]
                right = close[i]
                middle = close[j + 1:i]
                fractions = np.arange(1, span, dtype=float) / span
                chord = left + (right - left) * fractions
                if np.all(middle < chord):
                    visible_count += 1
            degree[i] = visible_count

        degree_series = pd.Series(degree, index=data.index)
        rolling_mean = degree_series.rolling(params.z_window, min_periods=params.z_window).mean()
        rolling_std = degree_series.rolling(params.z_window, min_periods=params.z_window).std(ddof=0)
        degree_z = (degree_series - rolling_mean) / rolling_std.replace(0.0, np.nan)
        lo = data["close"].rolling(params.position_window, min_periods=params.position_window).min()
        hi = data["close"].rolling(params.position_window, min_periods=params.position_window).max()
        relative_position = (data["close"] - lo) / (hi - lo).replace(0.0, np.nan)
        return pd.DataFrame({"degree_z": degree_z, "relative_position": relative_position}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        z = indicators["degree_z"]
        pos = indicators["relative_position"]
        topology_surge = z >= params.degree_z
        long_entry = topology_surge & (pos <= params.edge_fraction)
        short_entry = topology_surge & (pos >= 1.0 - params.edge_fraction)

        raw = np.zeros(len(data), dtype=int)
        state = 0
        for i, (go_long, go_short) in enumerate(zip(long_entry.fillna(False).to_numpy(), short_entry.fillna(False).to_numpy())):
            if state == 1:
                if go_short:
                    state = 0
            elif state == -1:
                if go_long:
                    state = 0
            elif go_long:
                state = 1
            elif go_short:
                state = -1
            raw[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
