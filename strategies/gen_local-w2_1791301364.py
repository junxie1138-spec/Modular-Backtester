from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    window: int = 24
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791301364"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @classmethod
    def warmup_bars(cls, params: RecoveryParams) -> int:
        return 2 * params.window + 2

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        pattern = (a.gt(b).astype(int) * 4 + a.gt(c).astype(int) * 2 + b.gt(c).astype(int))
        valid = a.notna() & b.notna() & c.notna()
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = pattern.eq(code).where(valid).rolling(params.window, min_periods=params.window).mean()
            entropy -= probability * np.log(probability.where(probability.gt(0), 1.0))
        entropy = entropy.fillna(0.0)
        baseline = entropy.where(entropy.gt(0)).rolling(params.window, min_periods=params.window).median().fillna(0.0)
        drawdown = close.div(close.rolling(params.window, min_periods=params.window).max()).sub(1.0).fillna(0.0)
        return pd.DataFrame({"entropy": entropy, "baseline": baseline, "drawdown": drawdown}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entropy = indicators["entropy"]
        baseline = indicators["baseline"]
        drawdown = indicators["drawdown"]
        depressed = drawdown.le(-0.025) & baseline.gt(0)
        long_entry = depressed & entropy.gt(baseline) & entropy.shift(1).le(baseline.shift(1)) & drawdown.diff().gt(0)
        short_entry = depressed & entropy.lt(baseline) & entropy.shift(1).ge(baseline.shift(1)) & drawdown.diff().lt(0)
        longs = long_entry.fillna(False).to_numpy()
        shorts = short_entry.fillna(False).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        for i in range(self.warmup_bars(params), len(data)):
            if position and i - entry_bar >= params.hold_bars:
                position = 0
            elif position == 0 and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
