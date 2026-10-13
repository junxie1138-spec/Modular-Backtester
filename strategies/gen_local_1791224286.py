from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryBreadthParams:
    peak_window: int = 12
    memory_window: int = 6
    breadth_window: int = 4
    min_up_days: int = 3
    drawdown_threshold: float = 0.035


class GeneratedStrategy(BaseStrategy[RecoveryBreadthParams]):
    strategy_id = "gen_local_1791224286"

    @classmethod
    def params_type(cls) -> type[RecoveryBreadthParams]:
        return RecoveryBreadthParams

    def warmup_bars(self, params: RecoveryBreadthParams) -> int:
        return max(params.peak_window + params.memory_window - 1, params.breadth_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryBreadthParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change()
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = close.div(peak).sub(1.0)
        stress = drawdown.rolling(params.memory_window, min_periods=params.memory_window).min()
        breadth = returns.gt(0).astype(float).rolling(params.breadth_window, min_periods=params.breadth_window).sum()
        return pd.DataFrame({"stress": stress.fillna(0.0), "breadth": breadth.fillna(0.0)}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryBreadthParams,
    ) -> SignalFrame:
        entry_condition = indicators["stress"].le(-params.drawdown_threshold) & indicators["breadth"].ge(params.min_up_days)
        df = pd.DataFrame(index=data.index)
        df["signal"] = np.where(entry_condition, 1, 0)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
