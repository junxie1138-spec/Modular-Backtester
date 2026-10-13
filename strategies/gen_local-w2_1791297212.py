from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LocalPositionParams:
    channel_bars: int = 8
    fast_span: int = 3
    slow_span: int = 8
    position_floor: float = 0.55


class GeneratedStrategy(BaseStrategy[LocalPositionParams]):
    strategy_id = "gen_local-w2_1791297212"

    @classmethod
    def params_type(cls) -> type[LocalPositionParams]:
        return LocalPositionParams

    @classmethod
    def warmup_bars(cls, params: LocalPositionParams) -> int:
        return max(params.channel_bars, params.slow_span) + 2

    def indicators(self, data: pd.DataFrame, params: LocalPositionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        upper = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max()
        lower = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min()
        width = (upper - lower).replace(0, np.nan)
        fast = data["close"].ewm(span=params.fast_span, adjust=False).mean()
        slow = data["close"].ewm(span=params.slow_span, adjust=False).mean()
        out["relative_position"] = ((data["close"] - lower) / width).fillna(0.5)
        out["slope_innovation"] = ((fast - slow).diff(2) / width).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: LocalPositionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        active = (indicators["slope_innovation"] > 0) & (indicators["relative_position"] > params.position_floor)
        df["signal"] = active.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
