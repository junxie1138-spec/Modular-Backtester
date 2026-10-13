from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_span: int = 9
    regime_window: int = 12
    jump_threshold: float = 0.25
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791297261"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return params.graph_span + params.regime_window

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data["close"]
        degree = pd.Series(0.0, index=data.index)
        for distance in range(1, params.graph_span):
            past = close.shift(distance)
            visible = past.notna()
            for offset in range(1, distance):
                line = close + (past - close) * offset / distance
                visible &= close.shift(offset) < line
            degree += visible.astype(float)

        degree = (degree / (params.graph_span - 1)).where(
            close.shift(params.graph_span - 1).notna(), 0.0
        )
        previous = degree.shift(1)
        out = pd.DataFrame(index=data.index)
        out["degree"] = degree
        out["jump"] = (degree - previous).fillna(0.0)
        out["previous"] = previous.fillna(0.0)
        out["prior_mean"] = previous.rolling(
            params.regime_window, min_periods=params.regime_window
        ).mean().fillna(0.0)
        out["ready"] = close.shift(
            params.graph_span + params.regime_window - 1
        ).notna()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        trigger = (
            indicators["ready"]
            & (indicators["previous"] < indicators["prior_mean"])
            & (indicators["jump"] >= params.jump_threshold)
            & (data["close"] > data["close"].shift(1))
        ).fillna(False)
        raw_signal = np.zeros(len(data), dtype=int)
        raw_size = np.ones(len(data), dtype=float)
        entry_bar = -1
        entry_size = 1.0

        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                    continue
                raw_signal[i] = 1
                raw_size[i] = entry_size
            elif trigger.iat[i]:
                entry_bar = i
                strength = (indicators["jump"].iat[i] - params.jump_threshold) / (1.0 - params.jump_threshold)
                entry_size = 0.35 + 0.60 * min(1.0, max(0.0, strength))
                raw_signal[i] = 1
                raw_size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(raw_size, index=data.index).shift(1).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
