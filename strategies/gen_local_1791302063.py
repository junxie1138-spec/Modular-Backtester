from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CandleRankParams:
    rank_window: int = 3
    extreme_cutoff: float = 0.25


class GeneratedStrategy(BaseStrategy[CandleRankParams]):
    strategy_id = "gen_local_1791302063"

    @classmethod
    def params_type(cls) -> type[CandleRankParams]:
        return CandleRankParams

    @classmethod
    def warmup_bars(cls, params: CandleRankParams) -> int:
        return params.rank_window

    def indicators(self, data: pd.DataFrame, params: CandleRankParams) -> pd.DataFrame:
        span = (data["high"] - data["low"]).replace(0, np.nan)
        body = ((data["close"] - data["open"]) / span).fillna(0.0)
        comparisons = [
            body.gt(body.shift(lag)).astype(float)
            + 0.5 * body.eq(body.shift(lag)).astype(float)
            for lag in range(1, params.rank_window)
        ]
        rank = sum(comparisons) / (params.rank_window - 1)
        rank.iloc[: params.rank_window - 1] = 0.5
        return pd.DataFrame({"body_rank": rank}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CandleRankParams,
    ) -> SignalFrame:
        rank = indicators["body_rank"]
        long_entry = rank.le(params.extreme_cutoff)
        short_entry = rank.ge(1.0 - params.extreme_cutoff)
        signal = []
        position = 0
        for go_long, go_short in zip(long_entry, short_entry):
            if position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            elif position == 0:
                if go_long:
                    position = 1
                elif go_short:
                    position = -1
            signal.append(position)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
