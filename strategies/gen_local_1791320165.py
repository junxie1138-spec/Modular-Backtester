from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 8
    reaction_window: int = 3
    baseline_window: int = 12
    shock_threshold: float = 0.22
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791320165"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.graph_window + params.reaction_window + params.baseline_window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 2 or params.reaction_window < 1 or params.baseline_window < 1:
            raise ValueError("Visibility windows must be positive; graph_window must be at least two.")
        if params.shock_threshold <= 0 or not 0 < params.minimum_size <= 1:
            raise ValueError("Require positive shock_threshold and minimum_size in (0, 1].")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        prices = out["log_close"]
        prior_min = (prices - prices.shift(1)).to_numpy()
        prior_max = prior_min.copy()
        upper_span = np.zeros(len(data), dtype=float)
        lower_span = np.zeros(len(data), dtype=float)

        # Natural visibility requires every intervening point to lie below
        # the connecting line; reflected visibility reverses that inequality.
        for lag in range(2, params.graph_window + 1):
            slope = ((prices - prices.shift(lag)) / lag).to_numpy()
            upper_span += lag * (slope < prior_min)
            lower_span += lag * (slope > prior_max)
            prior_min = np.minimum(prior_min, slope)
            prior_max = np.maximum(prior_max, slope)

        valid = prices.rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        normalizer = float(params.graph_window * (params.graph_window + 1) // 2 - 1)
        out["upper_span"] = pd.Series(upper_span, index=data.index).where(valid)
        out["lower_span"] = pd.Series(lower_span, index=data.index).where(valid)
        out["asymmetry"] = (out["upper_span"] - out["lower_span"]) / normalizer
        out["recent_topology"] = out["asymmetry"].rolling(params.reaction_window).mean()
        out["baseline_topology"] = out["asymmetry"].shift(params.reaction_window).rolling(params.baseline_window).mean()
        out["shock"] = out["recent_topology"] - out["baseline_topology"]
        out["strength"] = (out["shock"].abs() / (2.0 * params.shock_threshold)).clip(params.minimum_size, 1.0)
        out["ready"] = out[["recent_topology", "baseline_topology", "shock", "strength"]].notna().all(axis=1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        ready = indicators["ready"].fillna(False).astype(bool)
        shock = indicators["shock"].fillna(0.0)
        topology = indicators["recent_topology"].fillna(0.0)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[ready & shock.ge(params.shock_threshold) & topology.gt(0)] = 1
        entry.loc[ready & shock.le(-params.shock_threshold) & topology.lt(0)] = -1
        strength = indicators["strength"].fillna(params.minimum_size).clip(params.minimum_size, 1.0).to_numpy()

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        position = 0
        position_size = params.minimum_size
        for i, direction in enumerate(entry.to_numpy()):
            if position != 0 and direction == -position:
                position = 0
                position_size = params.minimum_size
            elif position == 0 and direction != 0:
                position = int(direction)
                position_size = float(strength[i])
            signals[i] = position
            sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"ready": True, "shock": [0.0, 0.44, 0.0, -0.44, -0.44, 0.0, 0.44], "recent_topology": [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5], "strength": [0.25, 1.0, 0.25, 1.0, 1.0, 0.25, 1.0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, VisibilityParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert result["size"].tolist() == [0.25, 0.25, 1.0, 1.0, 0.25, 1.0, 1.0]
    assert GeneratedStrategy().indicators(data, VisibilityParams())["ready"].eq(False).all()


if __name__ == "__main__":
    _self_check()
