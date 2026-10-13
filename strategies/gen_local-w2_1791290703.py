from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBridgeParams:
    visibility_window: int = 18
    atr_window: int = 14
    acceptance_window: int = 3
    graph_threshold: float = 0.35
    acceptance_threshold: float = 0.15
    min_degree: int = 3
    min_remote_edges: int = 2
    stop_atr: float = 2.5
    hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[GapBridgeParams]):
    strategy_id = "gen_local-w2_1791290703"

    @classmethod
    def params_type(cls) -> type[GapBridgeParams]:
        return GapBridgeParams

    @staticmethod
    def warmup_bars(params: GapBridgeParams) -> int:
        return int(max(params.visibility_window + 1,
                       params.atr_window + params.acceptance_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapBridgeParams) -> pd.DataFrame:
        if min(params.visibility_window, params.atr_window,
               params.acceptance_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive")
        previous_close = data["close"].shift(1)
        gap = (data["open"] / previous_close.where(previous_close > 0) - 1.0)
        magnitude = gap.abs()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window,
                                 min_periods=params.atr_window).mean()
        session_pressure = (data["close"] - data["open"]) / atr.where(atr > 0)
        acceptance = session_pressure.rolling(
            params.acceptance_window,
            min_periods=params.acceptance_window,
        ).mean()
        degree = pd.Series(0.0, index=data.index)
        remote_edges = pd.Series(0.0, index=data.index)
        signed_neighbors = pd.Series(0.0, index=data.index)
        blocker = pd.Series(-np.inf, index=data.index)
        for lag in range(1, params.visibility_window):
            historical_height = magnitude.shift(lag)
            visible = ((magnitude > blocker) &
                       (historical_height > blocker) &
                       magnitude.notna() & historical_height.notna())
            degree += visible.astype(float)
            signed_neighbors += np.sign(gap.shift(lag)).fillna(0.0) * visible
            if lag >= 3:
                remote_edges += visible.astype(float)
            blocker = pd.Series(
                np.maximum(blocker.to_numpy(), historical_height.to_numpy()),
                index=data.index,
            )
        graph_score = signed_neighbors / degree.clip(lower=1.0)
        ready = (gap.rolling(params.visibility_window).count().eq(params.visibility_window)
                 & atr.gt(0) & acceptance.notna())
        return pd.DataFrame({
            "gap": gap,
            "gap_magnitude": magnitude,
            "true_range": true_range,
            "atr": atr,
            "session_pressure": session_pressure,
            "acceptance": acceptance,
            "visibility_degree": degree,
            "remote_edges": remote_edges,
            "signed_neighbors": signed_neighbors,
            "graph_score": graph_score,
            "ready": ready,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapBridgeParams) -> SignalFrame:
        topology_valid = (indicators["ready"].fillna(False)
                          & indicators["visibility_degree"].ge(params.min_degree)
                          & indicators["remote_edges"].ge(params.min_remote_edges))
        long_topology = (topology_valid & indicators["gap"].gt(0)
                         & indicators["graph_score"].ge(params.graph_threshold))
        short_topology = (topology_valid & indicators["gap"].lt(0)
                          & indicators["graph_score"].le(-params.graph_threshold))
        long_acceptance = (indicators["session_pressure"].gt(0)
                           & indicators["acceptance"].ge(params.acceptance_threshold))
        short_acceptance = (indicators["session_pressure"].lt(0)
                            & indicators["acceptance"].le(-params.acceptance_threshold))
        entries = np.select([
            (long_topology & long_acceptance).fillna(False).to_numpy(dtype=bool),
            (short_topology & short_acceptance).fillna(False).to_numpy(dtype=bool),
        ], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        entry_bar = -1
        stop_price = np.nan
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and i > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0):
                    position = 0
                else:
                    entry_bar = i
                    stop_price = opens[i] - position * params.stop_atr * atrs[i - 1]
            if position:
                stopped = (np.isfinite(closes[i]) and
                           ((position == 1 and closes[i] < stop_price) or
                            (position == -1 and closes[i] > stop_price)))
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
            elif (entries[i] != 0 and np.isfinite(atrs[i]) and atrs[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                position = int(entries[i])
                pending_entry = True
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
