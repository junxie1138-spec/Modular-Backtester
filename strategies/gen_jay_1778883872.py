from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    rank_low: float = 0.20
    rank_high: float = 0.80
    contagion_low: float = 0.40
    contagion_high: float = 0.60


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778883872"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rank_window bars for rolling + 1 for the shift inside daily_up
        return params.rank_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        # Rolling percentile rank of close (0..1)
        ind["price_rank"] = (
            data["close"]
            .rolling(params.rank_window)
            .rank(pct=True)
        )

        # Contagion ratio: fraction of bars in window where close > prior close
        # Analogous to the infected population in an SI epidemic model
        daily_up = (data["close"] > data["close"].shift(1)).astype(float)
        ind["contagion"] = daily_up.rolling(params.rank_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        price_rank = indicators["price_rank"].values
        contagion = indicators["contagion"].values

        rank_low = params.rank_low
        rank_high = params.rank_high
        contagion_low = params.contagion_low
        contagion_high = params.contagion_high

        n = len(data)
        signal = np.zeros(n, dtype=np.int64)
        position = 0

        for i in range(n):
            pr = price_rank[i]
            ct = contagion[i]

            if np.isnan(pr) or np.isnan(ct):
                signal[i] = 0
                position = 0
                continue

            # Symmetric exit: exit long when rank crosses into short-entry zone
            #                 exit short when rank crosses into long-entry zone
            if position == 1 and pr > rank_high:
                position = 0
            elif position == -1 and pr < rank_low:
                position = 0

            # Entry: epidemic trough (depleted sellers) -> long
            #        epidemic peak (exhausted buyers)  -> short
            if position == 0:
                if pr < rank_low and ct < contagion_low:
                    position = 1
                elif pr > rank_high and ct > contagion_high:
                    position = -1

            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0

        # Mandatory one-bar shift: decision made at bar N close, filled at bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
