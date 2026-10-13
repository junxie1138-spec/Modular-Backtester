from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    recovery_window: int = 4
    min_drawdown: float = 0.025


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791301437"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if params.peak_window < 2 or params.recovery_window < 1:
            raise ValueError("Lookback windows must be positive.")
        if not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("min_drawdown must be between zero and one.")
        return int(params.peak_window + params.recovery_window)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        peak = data["high"].rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - data["close"] / peak).clip(lower=0.0)
        out["start_depth"] = depth.shift(params.recovery_window).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["recovery"] = (depth.shift(params.recovery_window) - depth).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gaps = data["open"] / data["close"].shift(1) - 1.0
        out["gap_pressure"] = gaps.rolling(params.recovery_window, min_periods=params.recovery_window).sum().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        gap_up = indicators["gap_pressure"].gt(0.0)
        recovering = indicators["recovery"].gt(0.0)
        entry = indicators["start_depth"].ge(params.min_drawdown) & gap_up & recovering
        reversal = ~gap_up | ~recovering
        signals = np.zeros(len(data), dtype=int)
        long = False
        for i, (enter, reverse) in enumerate(zip(entry.to_numpy(), reversal.to_numpy())):
            if long and reverse:
                long = False
            elif not long and enter:
                long = True
            signals[i] = int(long)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.RangeIndex(6)
    sample = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "start_depth": [0.0, 0.03, 0.03, 0.03, 0.03, 0.03],
        "gap_pressure": [0.0, 0.01, 0.01, 0.01, -0.01, 0.01],
        "recovery": [0.0, 0.01, 0.01, -0.01, 0.01, 0.01],
    }, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, RecoveryParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
