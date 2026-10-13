from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 4
    scale_window: int = 8
    monday_weight: float = 1.5
    threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791268193"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not (2 <= params.gap_window <= 9 and 2 <= params.scale_window <= 9):
            raise ValueError("Lookbacks must be between 2 and 9.")
        if not (np.isfinite(params.monday_weight) and params.monday_weight > 0
                and np.isfinite(params.threshold) and params.threshold > 0):
            raise ValueError("Weights and threshold must be finite and positive.")
        return int(max(params.gap_window, params.scale_window) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / previous - 1).replace([np.inf, -np.inf], np.nan)
        gap = gap.where(data["open"] > 0)
        weight = pd.Series(
            np.where(data.index.dayofweek == 0, -params.monday_weight, 1.0),
            index=data.index,
        )
        seasonal_gap = gap * weight
        pressure = seasonal_gap.rolling(params.gap_window, min_periods=params.gap_window).mean()
        scale = gap.abs().rolling(params.scale_window, min_periods=params.scale_window).mean()
        score = (pressure / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "gap": gap,
            "weekday_weight": weight,
            "seasonal_gap": seasonal_gap,
            "pressure": pressure,
            "scale": scale,
            "score": score,
        }, index=data.index).fillna(0.0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        score = indicators["score"].to_numpy()
        intent = np.where(score >= params.threshold, 1,
                          np.where(score <= -params.threshold, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, desired in enumerate(intent):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    data = pd.DataFrame(index=pd.date_range("2024-01-01", periods=8, freq="B"))
    indicators = pd.DataFrame({"score": [1, 1, 0, -1, -1, 0, 1, 1]}, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert (result.data["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(None, params) == 9


if __name__ == "__main__":
    _self_check()
