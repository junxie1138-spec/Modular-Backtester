from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownMemoryParams:
    peak_window: int = 63
    memory_window: int = 18
    minimum_memory_depth: float = 0.01
    recovery_fraction: float = 0.65


class GeneratedStrategy(BaseStrategy[DrawdownMemoryParams]):
    strategy_id = "gen_local_1790901054"

    @classmethod
    def params_type(cls):
        return DrawdownMemoryParams

    def warmup_bars(self, params: DrawdownMemoryParams) -> int:
        return max(2, int(params.peak_window)) + max(2, int(params.memory_window)) - 1

    def indicators(self, data: pd.DataFrame, params: DrawdownMemoryParams) -> pd.DataFrame:
        peak_window = max(2, int(params.peak_window))
        memory_window = max(2, int(params.memory_window))
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(peak_window, min_periods=peak_window).max()
        depth = (1.0 - close / peak.replace(0.0, np.nan)).clip(lower=0.0)
        memory = depth.rolling(memory_window, min_periods=memory_window).mean()
        return pd.DataFrame(
            {"peak": peak, "depth": depth, "depth_memory": memory},
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownMemoryParams,
    ) -> SignalFrame:
        condition = (
            indicators["depth"].notna()
            & indicators["depth_memory"].notna()
            & (indicators["depth_memory"] >= params.minimum_memory_depth)
            & (indicators["depth"] <= params.recovery_fraction * indicators["depth_memory"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Two consecutive qualifying closes enter; the first false condition exits.
        df["signal"] = (condition & condition.shift(1, fill_value=False)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DrawdownMemoryParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"depth": [0.08, 0.03, 0.02, 0.07, 0.03, 0.02, 0.01], "depth_memory": 0.06},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    indicators["depth_memory"] = np.nan
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
