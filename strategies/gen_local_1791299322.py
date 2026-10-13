from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    vol_window: int = 5
    spike_percentile: float = 0.85
    drawdown_percentile: float = 0.25


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791299322"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(83, params.vol_window + 70)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"]
        drawdown = close.div(close.rolling(20, min_periods=20).max()).sub(1)
        drawdown_cutoff = drawdown.rolling(63, min_periods=63).quantile(
            params.drawdown_percentile
        ).shift(1)

        volatility = close.pct_change().rolling(
            params.vol_window, min_periods=params.vol_window
        ).std().rolling(3, min_periods=3).mean()
        spike_cutoff = volatility.rolling(63, min_periods=63).quantile(
            params.spike_percentile
        ).shift(1)
        spike = volatility.gt(spike_cutoff)

        out = pd.DataFrame(index=data.index)
        out["deep_drawdown"] = drawdown.lt(drawdown_cutoff).fillna(False)
        out["recent_spike"] = spike.shift(1).rolling(4, min_periods=4).max().fillna(0).astype(bool)
        out["vol_easing"] = volatility.lt(volatility.shift(1)).fillna(False)
        out["vol_rising"] = volatility.gt(volatility.shift(1)).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        entry = (
            indicators["deep_drawdown"]
            & indicators["recent_spike"]
            & indicators["vol_easing"]
            & data["close"].gt(data["close"].shift(1))
        )
        exit_reversal = indicators["vol_rising"]
        position = 0
        signals = []
        for buy, sell in zip(entry, exit_reversal):
            if position and sell:
                position = 0
            elif not position and buy:
                position = 1
            signals.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
