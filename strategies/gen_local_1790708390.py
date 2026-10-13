from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    diff_order: float = 0.5
    diff_terms: int = 12
    position_window: int = 15
    percentile_window: int = 80
    tail_percentile: float = 0.2
    atr_window: int = 14
    stop_atr: float = 1.2
    hold_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790708390"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.diff_terms + params.position_window + params.percentile_window, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        weights = [1.0]
        for k in range(1, params.diff_terms):
            weights.append(-weights[-1] * (params.diff_order - k + 1) / k)
        fd = sum(w * close.shift(k) for k, w in enumerate(weights))
        minimum = fd.rolling(params.position_window, min_periods=params.position_window).min()
        maximum = fd.rolling(params.position_window, min_periods=params.position_window).max()
        position = ((fd - minimum) / (maximum - minimum).replace(0, np.nan)).clip(0, 1)
        history = position.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        previous_close = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        return pd.DataFrame({"position": position, "lower": history.quantile(params.tail_percentile), "upper": history.quantile(1 - params.tail_percentile), "atr": true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        position = indicators["position"].to_numpy(dtype=float)
        lower = indicators["lower"].to_numpy(dtype=float)
        upper = indicators["upper"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        entry_atr = 0.0
        stop = 0.0
        for i in range(1, len(data)):
            if side:
                if age == 0:
                    stop = opening[i] - side * params.stop_atr * entry_atr
                age += 1
                stopped = (side == 1 and close[i] <= stop) or (side == -1 and close[i] >= stop)
                if stopped or age >= params.hold_bars:
                    side = 0
                    target[i] = 0
                    continue
            if side == 0 and np.isfinite([position[i], lower[i], upper[i], position[i - 1], lower[i - 1], upper[i - 1], atr[i]]).all():
                if position[i] <= lower[i] and position[i - 1] > lower[i - 1]:
                    side = 1
                elif position[i] >= upper[i] and position[i - 1] < upper[i - 1]:
                    side = -1
                if side:
                    entry_atr = atr[i]
                    age = 0
            target[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
