from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthParams:
    channel_bars: int = 5
    budget_bars: int = 3
    threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[DepthParams]):
    strategy_id = "gen_local_1791228030"

    @classmethod
    def params_type(cls) -> type[DepthParams]:
        return DepthParams

    def warmup_bars(self, params: DepthParams) -> int:
        return int(params.channel_bars + params.budget_bars - 1)

    def indicators(self, data: pd.DataFrame, params: DepthParams) -> pd.DataFrame:
        if params.channel_bars < 2 or params.budget_bars < 1:
            raise ValueError("Lookbacks must be positive, with channel_bars >= 2")
        if not np.isfinite(params.threshold) or not 0 < params.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        log_close = np.log(data["close"].where(data["close"] > 0))
        log_close = log_close.replace([np.inf, -np.inf], np.nan)
        peak = log_close.rolling(params.channel_bars, min_periods=params.channel_bars).max()
        trough = log_close.rolling(params.channel_bars, min_periods=params.channel_bars).min()
        down_depth = (peak - log_close).clip(lower=0)
        up_depth = (log_close - trough).clip(lower=0)
        down_budget = down_depth.pow(2).rolling(params.budget_bars, min_periods=params.budget_bars).mean()
        up_budget = up_depth.pow(2).rolling(params.budget_bars, min_periods=params.budget_bars).mean()
        total_budget = up_budget + down_budget
        strength = ((up_budget - down_budget) / total_budget.where(total_budget > 0)).fillna(0.0)
        return pd.DataFrame({
            "log_close": log_close,
            "peak": peak,
            "trough": trough,
            "down_depth": down_depth,
            "up_depth": up_depth,
            "down_budget": down_budget,
            "up_budget": up_budget,
            "total_budget": total_budget,
            "strength": strength,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthParams) -> SignalFrame:
        strength = indicators["strength"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        entry = pd.Series(np.where(strength >= params.threshold, 1, np.where(strength <= -params.threshold, -1, 0)), index=data.index)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DepthParams()
    data = pd.DataFrame({"close": [100.0] * 7}, index=pd.date_range("2020-01-01", periods=7))
    indicators = pd.DataFrame({"strength": [0.5, 0.0, -0.5, -0.5, 0.0, 0.5, 0.5]}, index=data.index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert (strategy.indicators(data, params)["strength"] == 0).all()
    path = pd.DataFrame({"close": [100., 101., 102., 103., 102., 104., 105., 104., 103., 102., 101., 100.]})
    mirror = pd.DataFrame({"close": 1.0 / path["close"]})
    assert np.allclose(strategy.indicators(path, params)["strength"], -strategy.indicators(mirror, params)["strength"])
