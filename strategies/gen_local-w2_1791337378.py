from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownShiftParams:
    window: int = 18
    depth_shift: float = 0.01


class GeneratedStrategy(BaseStrategy[DrawdownShiftParams]):
    strategy_id = "gen_local-w2_1791337378"

    @classmethod
    def params_type(cls) -> type[DrawdownShiftParams]:
        return DrawdownShiftParams

    def warmup_bars(self, params: DrawdownShiftParams) -> int:
        return int(126 + 2 * params.window)

    def indicators(self, data: pd.DataFrame, params: DrawdownShiftParams) -> pd.DataFrame:
        if params.window < 1 or not np.isfinite(params.depth_shift) or params.depth_shift <= 0:
            raise ValueError("window and depth_shift must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        out["peak"] = close.rolling(126, min_periods=126).max()
        out["drawdown"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["recent_depth"] = out["drawdown"].rolling(params.window, min_periods=params.window).mean()
        out["previous_depth"] = out["recent_depth"].shift(params.window)
        out["depth_change"] = out["recent_depth"] - out["previous_depth"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownShiftParams) -> SignalFrame:
        change = indicators["depth_change"]
        events = np.select(
            [change.le(-params.depth_shift), change.ge(params.depth_shift)],
            [1, -1],
            default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DrawdownShiftParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"depth_change": [np.nan, -0.02, 0.0, -0.02, 0.02, 0.02, 0.0, -0.02, -0.02]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()
