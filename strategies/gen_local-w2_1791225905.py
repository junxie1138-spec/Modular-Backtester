from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapWaveParams:
    lookback: int = 20
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[GapWaveParams]):
    strategy_id = "gen_local-w2_1791225905"

    @classmethod
    def params_type(cls) -> type[GapWaveParams]:
        return GapWaveParams

    def warmup_bars(self, params: GapWaveParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: GapWaveParams) -> pd.DataFrame:
        gap = data["open"].div(data["close"].shift(1)).sub(1)
        autocorr = gap.rolling(params.lookback, min_periods=params.lookback).corr(gap.shift(1))
        return pd.DataFrame({
            "gap": gap.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "autocorr": autocorr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapWaveParams,
    ) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        gaps = indicators["gap"].to_numpy()
        autocorr = indicators["autocorr"].to_numpy()
        position = 0
        entry_price = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if i and desired[i - 1] != position:
                position = desired[i - 1]
                if position:
                    entry_price = opens[i]
                    entry_bar = i

            if position:
                gain = position * (closes[i] / entry_price - 1)
                desired[i] = 0 if gain >= params.profit_target or i - entry_bar + 1 >= 2 else position
            elif i >= self.warmup_bars(params) and abs(gaps[i]) >= 0.002 and abs(autocorr[i]) >= 0.2:
                desired[i] = 1 if autocorr[i] * gaps[i] > 0 else -1

        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
