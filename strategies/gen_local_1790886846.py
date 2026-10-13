from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    baseline_window: int = 30
    atr_window: int = 14
    topology_z: float = 0.8
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790886846"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.graph_window + params.baseline_window + 2,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if (params.graph_window < 3 or params.baseline_window < 2
                or params.atr_window < 1 or params.hold_bars not in (1, 2)
                or not np.isfinite(params.topology_z) or params.topology_z <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid visibility strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["magnitude"] = out["return"].abs()
        width = params.graph_window
        heights = np.column_stack([
            out["magnitude"].shift(width - 1 - j).to_numpy()
            for j in range(width)
        ])
        degrees = np.zeros_like(heights)
        valid = np.isfinite(heights).all(axis=1)
        # Horizontal visibility: every intervening height must be below both endpoints.
        for left in range(width - 1):
            obstruction = np.full(len(data), -np.inf)
            for right in range(left + 1, width):
                edge = (np.minimum(heights[:, left], heights[:, right])
                        > obstruction) & valid
                degrees[:, left] += edge
                degrees[:, right] += edge
                obstruction = np.maximum(obstruction, heights[:, right])
        for j in range(width):
            out[f"degree_{j}"] = np.where(valid, degrees[:, j], np.nan)
        total = degrees.sum(axis=1)
        out["hub_share"] = np.divide(
            degrees.max(axis=1), total,
            out=np.full(len(data), np.nan), where=valid & (total > 0))
        history = out["hub_share"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window)
        out["hub_mean"] = history.mean()
        out["hub_std"] = history.std(ddof=0)
        out["topology_z"] = ((out["hub_share"] - out["hub_mean"])
                             / out["hub_std"].replace(0.0, np.nan))
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams,
    ) -> SignalFrame:
        qualified = (
            np.isfinite(indicators["topology_z"])
            & indicators["topology_z"].ge(params.topology_z)
            & np.isfinite(indicators["atr"]) & indicators["atr"].gt(0)
        )
        side = np.sign(indicators["return"].fillna(0))
        candidate = side.where(qualified, 0).astype(int)
        confirmed = (candidate.ne(0)
                     & candidate.eq(candidate.shift(1)))
        # One entry opportunity per uninterrupted two-bar confirmation episode.
        entry = candidate.where(
            confirmed & ~confirmed.shift(1, fill_value=False), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        stop = np.nan
        pending_atr = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = pending
                    stop = opens[i] - position * params.stop_atr * pending_atr
                    age = 0
                pending = 0
            if position:
                age += 1
                stopped = (not np.isfinite(closes[i])
                           or (position == 1 and closes[i] < stop)
                           or (position == -1 and closes[i] > stop))
                if stopped or age >= params.hold_bars:
                    position = 0
                decisions[i] = position
                continue
            if entry[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(entry[i])
                pending_atr = atr[i]
                decisions[i] = pending
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 106.0, "low": 94.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "return": 0.01, "topology_z": 2.0,
        "atr": [2.0, 2.0, 100.0, 2.0, 2.0, 2.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 95.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    ind["return"] = -0.01
    data.loc[index[2], "close"] = 105.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
