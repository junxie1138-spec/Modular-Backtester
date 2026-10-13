from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 15
    topology_threshold: float = 0.65
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791302203"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @classmethod
    def warmup_bars(cls, params: VisibilityParams) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data["close"]
        degree = pd.Series(0, index=data.index, dtype=int)
        lower = pd.Series(0, index=data.index, dtype=int)
        barrier = pd.Series(-np.inf, index=data.index)

        for lag in range(1, params.window):
            earlier = close.shift(lag)
            visible = earlier.notna() & (np.minimum(close, earlier) > barrier)
            degree += visible.astype(int)
            lower += (visible & earlier.lt(close)).astype(int)
            barrier = pd.concat([barrier, earlier], axis=1).max(axis=1)

        low = close.rolling(params.window, min_periods=params.window).min()
        high = close.rolling(params.window, min_periods=params.window).max()
        position = ((close - low) / (high - low).replace(0, np.nan)).fillna(-1.0)
        score = (lower / degree.replace(0, np.nan)).fillna(0.0)
        return pd.DataFrame({"lower": lower, "degree": degree, "position": position, "score": score}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        eligible = (
            indicators["position"].between(0.25, 0.75)
            & indicators["lower"].ge(2)
            & indicators["lower"].gt(indicators["lower"].shift(1).fillna(0))
            & indicators["score"].ge(params.topology_threshold)
        ).to_numpy(dtype=bool)
        strength = indicators["score"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.35, dtype=float)
        exit_at = -1
        entry_size = 0.35

        for i in range(len(data)):
            if i == exit_at:
                exit_at = -1
            elif i < exit_at:
                signal[i] = 1
                size[i] = entry_size
            elif eligible[i]:
                entry_size = float(np.clip(0.35 + 0.60 * strength[i], 0.35, 0.95))
                signal[i] = 1
                size[i] = entry_size
                exit_at = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
