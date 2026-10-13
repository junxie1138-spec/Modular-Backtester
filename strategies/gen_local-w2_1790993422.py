from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 14
    atr_period: int = 14
    closure_threshold: float = 0.50
    impulse_atr: float = 0.35
    stop_atr: float = 1.20
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1790993422"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(max(max(3, params.graph_window) + 1,
                       max(2, params.atr_period) + 1, 3) + 1)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        window = max(3, int(params.graph_window))
        period = max(2, int(params.atr_period))
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        shock = np.log(close.where(close > 0)).diff().abs()
        nodes = np.column_stack([
            shock.shift(lag).to_numpy() for lag in range(window - 1, -1, -1)
        ])
        valid = np.isfinite(nodes).all(axis=1)
        adjacency = np.zeros((len(data), window, window), dtype=float)
        # Horizontal visibility: all intervening shocks lie below both endpoints.
        for left in range(window - 1):
            barrier = np.full(len(data), -np.inf)
            for right in range(left + 1, window):
                if right > left + 1:
                    barrier = np.maximum(barrier, nodes[:, right - 1])
                edge = valid & (np.minimum(nodes[:, left], nodes[:, right]) > barrier)
                adjacency[:, left, right] = edge
                adjacency[:, right, left] = edge
        degree = adjacency.sum(axis=2)
        wedges = (degree * (degree - 1)).sum(axis=1) / 2.0
        triangles = np.einsum("nij,nji->n", adjacency @ adjacency, adjacency) / 6.0
        closure = np.divide(3.0 * triangles, wedges,
                            out=np.zeros(len(data)), where=wedges > 0)
        closure[~valid] = np.nan
        out = pd.DataFrame(index=data.index)
        out["shock"] = shock
        out["atr"] = atr
        out["closure"] = closure
        out["impulse"] = (close - close.shift(2)) / atr.where(atr > 0)
        ready = (np.isfinite(out["closure"]) & np.isfinite(out["impulse"])
                 & np.isfinite(out["atr"]) & (out["atr"] > 0)
                 & np.isfinite(close) & (close > 0))
        out["regime"] = np.where(
            ready, np.where(out["closure"] >= params.closure_threshold, -1, 1), 0
        ).astype(int)
        direction = np.sign(out["impulse"].fillna(0)).astype(int)
        out["candidate"] = np.where(
            ready & (out["impulse"].abs() >= max(0.0, params.impulse_atr)),
            direction * out["regime"], 0
        ).astype(int)
        out["confirmed"] = np.where(
            (out["candidate"] != 0)
            & (out["candidate"] == out["candidate"].shift(1)),
            out["candidate"], 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TopologyParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        frozen_atr = np.nan
        age = 0
        horizon = min(2, max(1, int(params.holding_bars)))
        stop_multiple = max(0.01, float(params.stop_atr))
        for i in range(len(data)):
            exited = False
            if position != 0:
                # The previous close's entry decision fills at today's open.
                if not np.isfinite(entry):
                    entry = opens[i]
                age += 1
                invalid = not (np.isfinite(entry) and entry > 0
                               and np.isfinite(close[i]) and close[i] > 0)
                stopped = (
                    not invalid and
                    ((position == 1 and close[i] < entry - stop_multiple * frozen_atr)
                     or (position == -1 and close[i] > entry + stop_multiple * frozen_atr))
                )
                if invalid or stopped or age >= horizon:
                    position = 0
                    entry = np.nan
                    age = 0
                    exited = True
            if position == 0 and not exited and confirmed[i] != 0:
                if np.isfinite(atr[i]) and atr[i] > 0:
                    position = int(confirmed[i])
                    frozen_atr = atr[i]
                    entry = np.nan
                    age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
